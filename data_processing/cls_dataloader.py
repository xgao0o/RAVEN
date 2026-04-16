import pickle
import numpy as np
from glob import glob
import sys
import pandas as pd
import shelve
import pickle
from scipy.sparse import vstack
from scipy.sparse import hstack
import numpy as np
import scipy
import pandas as pd
from scipy.sparse import lil_matrix
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pickle
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import BertForPreTraining, BertTokenizer, BertConfig

from transformers.optimization import Adafactor, AdafactorSchedule
from tqdm import tqdm
import datetime
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import os
from scipy.sparse import csr_array, vstack, hstack
import matplotlib.pyplot as plt
import shelve
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
MASK_ID = 3
UNK_ID = 4
headers_special = ['PAD_ID', 'SEP_ID', 'CLS_ID', 'MASK_ID', 'UNK_ID']


class SeqCLSDataset(Dataset):
    def __init__(self, mode='train', n_tokens=57735, max_length=512, min_num_visits=1, model='BERT', folder="./", age=None, random_window=True):
        self.folder = folder
        files = sorted(glob(self.folder + "data_{}*.shelve.dat".format(mode)))
        self.keys, self.feats, self.labels = [], [], []
        self.db_dict = {}
        self.data_list = []
        for i, file in tqdm(enumerate(files)):
            self.db_dict[i] = shelve.open(file.strip(".dat"))
        for k, v in self.db_dict.items():
            for key in v.keys():
                self.data_list.append((k, key))
                    
        self.max_length = max_length
        self.min_num_visits = min_num_visits
        self.folder =  folder
        self.n_tokens = n_tokens + 5
        self.age = age
        self.random_window = random_window
        self.model = model
        if self.age is not None:
            print("USE AGE as positional embedding")
                
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        k, key = self.data_list[idx]
        datum = self.db_dict[k][key]
        a = datum[1]
        # a[:, [16918, 16919, 16920]] = 0
        label = datum[2]
        column_to_delete = 51235 # gives the day of the visit
        days_data = a[:, column_to_delete].toarray().astype("int").squeeze()
        #need to get the days and tokens in order
        days_arr , token_arr = np.nonzero(a)
        delete_idx = (token_arr != column_to_delete)
        days_arr = days_arr[delete_idx]
        token_arr = token_arr[delete_idx] + 5
        token_arr_raw = token_arr
        pos_embed_days = days_data[days_arr]

        change_indices = np.where(np.diff(pos_embed_days) != 0)[0] + 1
        pos_embed_days_final = np.insert(pos_embed_days, change_indices, np.array(days_data)[1:])
        att_days = np.insert(pos_embed_days, change_indices, np.array(days_data)[1:] - 1)
        token_list_final = np.insert(token_arr, change_indices, SEP_ID)
        change_indices = np.where(np.diff(pos_embed_days_final) != 0)[0] + 1
        

        starting_indices = np.concatenate(([0], change_indices))
        starting_indices[1:] = starting_indices[1:]+1
        if self.random_window:
            start_ind = np.random.randint(0,high=starting_indices.shape[0]-self.min_num_visits+1)
        else:
            start_ind = 0
        end_arr_fin = starting_indices - starting_indices[start_ind]
        end_ind = np.where(end_arr_fin < self.max_length)[0][-1]
        
        
        if (start_ind != end_ind):
            days_embed_arr = pos_embed_days_final[starting_indices[start_ind]:starting_indices[end_ind]]
            att_days_embed_arr = att_days[starting_indices[start_ind]:starting_indices[end_ind]]
            token_arr = token_list_final[starting_indices[start_ind]:starting_indices[end_ind]]
            cons_visits = np.arange(start_ind+1, end_ind+1)
        else:
            days_embed_arr = pos_embed_days_final[starting_indices[start_ind]:starting_indices[end_ind+1]][-self.max_length:]
            att_days_embed_arr = att_days[starting_indices[start_ind]:starting_indices[end_ind+1]][-self.max_length:]
            token_arr = token_list_final[starting_indices[start_ind]:starting_indices[end_ind+1]]
            sampled_elements = np.random.choice(token_arr[:-1], self.max_length - 1, replace=False)

            # Combine the sampled elements with the last element to create the new num_arr
            token_arr = np.concatenate([sampled_elements, [token_arr[-1]]])
            cons_visits = np.arange(start_ind+1,start_ind+2)
        pred_mask = (token_arr == SEP_ID)
        
        days_embed_arr = days_embed_arr - days_embed_arr[0]
    
        att_days_embed_arr = att_days_embed_arr - att_days_embed_arr[0]
        
        encounter_pos_embed = np.cumsum(np.diff(days_embed_arr, prepend=0) > 0)
        curr_size = len(token_arr)
        days_embed_arr = np.concatenate(([0], days_embed_arr[:-1]))
        encounter_pos_embed = np.concatenate(([0], encounter_pos_embed[:-1]))
        attn_mask = np.concatenate(([1], np.pad(np.ones((curr_size)), (0,self.max_length - curr_size), mode='constant', constant_values=PAD_ID)))
        token_arr = np.concatenate(([CLS_ID], np.pad(token_arr,(0,self.max_length - curr_size), mode='constant', constant_values=PAD_ID)))
        days_embed_arr = np.concatenate(([PAD_ID], np.pad(days_embed_arr + 1,(0,self.max_length - curr_size), mode='constant', constant_values=PAD_ID)))
        encounter_pos_embed = np.concatenate(([PAD_ID], np.pad(encounter_pos_embed + 1,(0,self.max_length - curr_size), mode='constant', constant_values=PAD_ID)))
        datum = {
            'input_ids': token_arr,
            'attention_mask':torch.from_numpy(attn_mask).to(torch.int64), 
            'token_type_ids':torch.from_numpy(encounter_pos_embed).to(torch.int64),
            'position_ids':torch.from_numpy(days_embed_arr).to(torch.int64),
            'labels':torch.from_numpy(np.array([label])).to(torch.int64)
        }
        return datum