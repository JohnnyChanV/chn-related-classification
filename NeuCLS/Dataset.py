import concurrent.futures
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import string
import os
import json
import torch
import pickle
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from transformers import AutoTokenizer


class RawDataset():

    def __init__(self,path):
        self.path = path
        self.raw = list(json.load(open(self.path,'r',encoding='utf-8')))

        indexes = [item['index'] for item in self.raw]
        texts = []
        labels = []
        print(f"[INFO]:DATA PROCESSING..{path}")
        for item in (self.raw):
            if item['vi_title']!=False and item['vi_body']!=False:
                texts.append(item['vi_title']+item['vi_body'])
                labels.append(item['label'])
        print("[index,words,label]")
        self.data = list(zip(indexes,texts,labels))
        self.data_train, self.data_test = train_test_split(self.data, test_size=0.3, random_state=42)
        # print(f"[INFO]: TOTAL DATA SIZE:{len(self.data)}")


    def get_train_test_data(self):
        print(f"[INFO]:TRAIN DATA SIZE:{len(self.data_train)}, TEST DATA SIZE:{len(self.data_test)}")
        return self.data_train, self.data_test

    def get_data(self):
        return self.data

    def __len__(self):
        return len(self.data)


class TorchDataset(data.Dataset):
##Data Preprocess for Classification

    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data
        self.num_worker = args.num_worker
        self.tokenizer = AutoTokenizer.from_pretrained(args.LLM)
        self.process_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        indexes,input_ids,mask,labels = self.data[index]
        indexes = torch.tensor([indexes],dtype=torch.long)
        input_ids = torch.tensor([input_ids],dtype=torch.long)
        mask = torch.tensor([mask],dtype=torch.long)
        labels = torch.tensor([labels],dtype=torch.long)

        return indexes,input_ids,mask,labels


    def process_item(self,item):

        text_encode_plus = self.tokenizer.encode_plus(item[1], truncation=True, padding='max_length',
                                                      max_length=self.args.max_length)

        i = (int(item[0]),
             text_encode_plus['input_ids'],
             text_encode_plus['attention_mask'],
             1 if item[-1] else 0)
        return i
    def process_data(self):
        dt = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_worker) as executor:
            for item in tqdm(self.data):
                future = executor.submit(self.process_item, item)
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures),total=len(self.data)):
                result = future.result()
                dt.append(result)
        self.data = dt
def collate_fn(X):
    X = list(zip(*X))
    # print(X)
    indexes,input_ids,mask,labels = X

    indexes = torch.cat(indexes, 0)
    input_ids = torch.cat(input_ids, 0)
    mask = torch.cat(mask, 0)
    labels = torch.cat(labels, 0)

    return indexes,input_ids,mask,labels
def data_loader(data_, args, shuffle=True, num_workers=0):
    dataset = TorchDataset(args, data_)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             pin_memory=False,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader


