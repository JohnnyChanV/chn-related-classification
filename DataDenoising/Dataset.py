import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import string

class Dataset():


    def __init__(self,path):
        self.path = path
        self.raw = list(json.load(open(self.path,'r',encoding='utf-8')))

        indexes = [item['index'] for item in self.raw]
        texts = []
        labels = []
        print("[INFO]:DATA PROCESSING..")
        for item in tqdm(self.raw):
            if item['vi_title']!=False and item['vi_body']!=False:
                texts.append(item['vi_title']+item['vi_body'])
                labels.append(item['label'])
        print("[index,words,label]")
        self.data = list(zip(indexes,texts,labels))
        self.data_train, self.data_test = train_test_split(self.data, test_size=0.2, random_state=42)
        # print(f"[INFO]: TOTAL DATA SIZE:{len(self.data)}")


    def get_train_test_data(self):
        print(f"[INFO]:TRAIN DATA SIZE:{len(self.data_train)}, TEST DATA SIZE:{len(self.data_test)}")
        return self.data_train, self.data_test

    def get_data(self):
        return self.data

    def __len__(self):
        return len(self.data)
