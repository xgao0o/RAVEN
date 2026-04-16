"""
Poor Man's Configurator. Example usage:
$ python scripts/train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

Usage from scripts:
    from raven.configurator import apply_overrides
    apply_overrides(globals())
"""

import sys
from ast import literal_eval


def apply_overrides(global_dict):
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read(), global_dict)
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=', 1)
            key = key[2:]
            if key in global_dict:
                try:
                    # attempt to eval it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(global_dict[key])
                print(f"Overriding: {key} = {attempt}")
                global_dict[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")
