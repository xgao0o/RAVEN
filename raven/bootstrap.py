"""Patient-clustered confidence intervals.

Draw patients with replacement and retain all windows from each draw. Metrics
remain window-based; only the resampling unit changes. Positive-patient strata
contain patients with at least one positive window. Thresholds stay fixed.
"""

from pathlib import Path
import shelve

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def _patient_groups(labels, patient_ids):
    labels = np.asarray(labels).reshape(-1)
    ids = np.asarray(patient_ids)
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels must be a nonempty binary array')
    if ids.ndim != 1 or len(ids) != len(labels) or pd.isna(ids).any():
        raise ValueError('Provide one nonmissing patient identifier per prediction row')
    ids = ids.astype(str)
    if np.any(np.char.strip(ids) == ''):
        raise ValueError('Patient identifiers cannot be empty')
    unique, inverse = np.unique(ids, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    counts = np.bincount(inverse, minlength=len(unique))
    start = np.zeros(len(unique), dtype=np.int64)
    start[1:] = np.cumsum(counts)[:-1]
    positive = np.zeros(len(unique), dtype=bool)
    np.logical_or.at(positive, inverse, labels.astype(bool))
    return labels, dict(order=order, counts=counts, start=start,
                        positive=np.flatnonzero(positive), negative=np.flatnonzero(~positive))


def _seeds(n_bootstraps, bootstrap_type, random_seed):
    if not isinstance(n_bootstraps, (int, np.integer)) or n_bootstraps < 1:
        raise ValueError('n_bootstraps must be a positive integer')
    if bootstrap_type not in ('stratified', 'standard'):
        raise ValueError('bootstrap_type must be stratified or standard')
    return np.random.RandomState(random_seed).randint(0, 2**31, size=n_bootstraps)


def _draw_rows(groups, bootstrap_type, seed):
    rng = np.random.RandomState(seed)
    if bootstrap_type == 'stratified':
        drawn = np.concatenate([
            rng.choice(groups['positive'], size=len(groups['positive']), replace=True),
            rng.choice(groups['negative'], size=len(groups['negative']), replace=True),
        ]).astype(np.int64)
    else:
        patients = len(groups['counts'])
        drawn = rng.randint(0, patients, size=patients)
    lengths = groups['counts'][drawn]
    segments = np.repeat(np.arange(len(drawn)), lengths)
    within = np.arange(int(lengths.sum())) - (np.cumsum(lengths) - lengths)[segments]
    return groups['order'][groups['start'][drawn][segments] + within]


def patient_bootstrap_indices(labels, patient_ids, n_bootstraps=1000,
                              bootstrap_type='stratified', random_seed=42):
    """Yield row indices for complete patient-cluster draws (also for curves)."""
    _, groups = _patient_groups(labels, patient_ids)
    for seed in _seeds(n_bootstraps, bootstrap_type, random_seed):
        yield _draw_rows(groups, bootstrap_type, int(seed))


def compute_metrics(labels, scores, threshold=None):
    result = {}
    if threshold is not None:
        predicted = scores >= threshold
        result.update(precision=precision_score(labels, predicted, zero_division=0),
                      recall=recall_score(labels, predicted, zero_division=0),
                      f1=f1_score(labels, predicted, zero_division=0))
    if len(np.unique(labels)) > 1:
        result.update(auroc=roc_auc_score(labels, scores),
                      auprc=average_precision_score(labels, scores))
    else:
        result.update(auroc=np.nan, auprc=np.nan)
    return result


def _bootstrap_metrics(scores, labels, groups, threshold, bootstrap_type, seed):
    rows = _draw_rows(groups, bootstrap_type, seed)
    return compute_metrics(labels[rows], scores[rows], threshold)


def patient_bootstrap_ci(scores, labels, patient_ids, threshold=None, n_bootstraps=1000,
                         bootstrap_type='stratified', confidence_level=0.95,
                         random_seed=42, n_jobs=1):
    """Return (metric intervals, patient count, positive-patient count).

    Point estimates use all original rows. Percentile intervals exclude undefined
    single-class replicates; ``n_valid`` reports the number used for each metric.
    Real patient identifiers are required, not model patient-position embeddings.
    """
    labels, groups = _patient_groups(labels, patient_ids)
    scores = np.asarray(scores).reshape(-1)
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and aligned with labels and patient identifiers')
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level must lie between zero and one')
    if threshold is not None and np.isnan(threshold):
        raise ValueError('threshold cannot be NaN')
    seeds = _seeds(n_bootstraps, bootstrap_type, random_seed)
    boot = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_bootstrap_metrics)(scores, labels, groups, threshold, bootstrap_type, int(seed))
        for seed in seeds
    )
    point = compute_metrics(labels, scores, threshold)
    alpha = 1 - confidence_level
    ci = {}
    for metric in point:
        values = np.array([sample[metric] for sample in boot])
        values = values[np.isfinite(values)]
        ci[metric] = dict(
            point_estimate=point[metric],
            mean=float(values.mean()) if len(values) else np.nan,
            std=float(values.std()) if len(values) else np.nan,
            lower=float(np.percentile(values, 100 * alpha / 2)) if len(values) else np.nan,
            upper=float(np.percentile(values, 100 * (1 - alpha / 2))) if len(values) else np.nan,
            n_valid=len(values),
        )
    return ci, len(groups['counts']), len(groups['positive'])


def align_legacy_predictions(scores, labels, data_folder, *, layout, world_size,
                             batch_size=8, n_gather=100):
    """Recover patient IDs for full-cohort score files without saved identifiers.

    Explicit layouts: ``legacy_periodic`` is the rebuttal evaluator's native
    shelve key order, padded sampler, and periodic rank gathers;
    ``rank_concatenated`` is the optimized evaluator's sorted key order, disjoint
    shards, and one final rank gather. Only full cohorts (num_patients=-1) are
    supported. All reconstructed labels are checked. Padding copies are removed
    from the CI input, as in the rebuttal code; evaluation sampling is unchanged.
    """
    if layout not in ('legacy_periodic', 'rank_concatenated'):
        raise ValueError('Specify legacy_periodic or rank_concatenated alignment')
    if min(world_size, batch_size, n_gather) <= 0:
        raise ValueError('world_size, batch_size, and n_gather must be positive')
    scores, labels = np.asarray(scores).reshape(-1), np.asarray(labels).reshape(-1)
    if len(scores) != len(labels):
        raise ValueError('Scores and labels must have the same length')
    paths = sorted({str(path).removesuffix('.dat') for pattern in
                    ('data_test*.shelve.dat', 'data_test*.shelve')
                    for path in Path(data_folder).glob(pattern)})
    keys, source_labels = [], []
    for path in paths:
        with shelve.open(path, flag='r') as db:
            ordered = sorted(db.keys()) if layout == 'rank_concatenated' else list(db.keys())
            for key in ordered:
                keys.append(key)
                source_labels.append(int(db[key][2]))
    count = len(keys)
    if not count:
        raise ValueError('No test cohort records found for legacy alignment')
    indices = np.arange(count)
    if layout == 'legacy_periodic':
        indices = np.resize(indices, ((count + world_size - 1) // world_size) * world_size)
    shards = [indices[rank::world_size] for rank in range(world_size)]
    if layout == 'legacy_periodic':
        chunk = batch_size * n_gather
        order = np.concatenate([shard[start:start + chunk]
                                for start in range(0, len(shards[0]), chunk) for shard in shards])
    else:
        order = np.concatenate(shards)
    if len(order) != len(labels) or not np.array_equal(np.asarray(source_labels)[order], labels):
        raise ValueError('Legacy prediction alignment failed: check the cohort, layout, and run settings; '
                         'shuffled/subsampled legacy runs require explicitly saved identifiers')
    _, first = np.unique(order, return_index=True)
    keep = np.sort(first)
    ids = np.asarray([keys[index].split('_', 1)[0] for index in order[keep]], dtype=str)
    _patient_groups(labels[keep], ids)
    return scores[keep], labels[keep], ids
