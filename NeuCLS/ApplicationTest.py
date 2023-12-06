import concurrent.futures

import pymysql
from pprint import pprint
import torch
from transformers import AutoTokenizer
class classifier():
    def __init__(self,device='mps'):
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.device = device
        self.model = torch.load('best_model.pt',map_location=torch.device(device))
    def encode(self,text):
        input_ids = self.tokenizer.encode_plus(text, truncation=True, padding='max_length',max_length=128,return_tensors='pt')['input_ids']
        return input_ids

    def process_data(self,text_set):
        batch_ids = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for text in text_set:
                future = executor.submit(self.encode, text)
                futures.append(future)

            for future in (concurrent.futures.as_completed(futures)):
                result = future.result()
                batch_ids.append(result)

        batch_ids = torch.cat(batch_ids,0)
        return batch_ids

    def classify(self,text_set):
        batch_ids = self.process_data(text_set)
        logits = self.model(batch_ids.to(self.device),None)
        proba = torch.softmax(logits, -1)
        print(proba)
        y_pred = proba.argmax(dim=-1).flatten().tolist()
        return y_pred

class database_handler():
    def __init__(self,batch_size = 4):
        self.batch_size = batch_size
        # dg_connect = pymysql.connect(host='',  # 本地数据库
        #                              user='',
        #                              password='',
        #                              charset='utf8')  # 服务器名,账户,密码，数据库名称
        self.dg_cur = dg_connect.cursor()
        self.dg_cur.execute("select count(*) from dg_crawler.news;")
        self.news_collection_size = int(self.dg_cur.fetchall()[0][0])
        self.current_pointer = 0

        # self.annotation_connet = pymysql.connect(host='',  # 本地数据库
        #                              user='',
        #                              password='',
        #                              charset='utf8')  # 服务器名,账户,密码，数据库名称
        self.annotation_cur = self.annotation_connet.cursor()

    def get_data(self):
        if self.current_pointer < self.news_collection_size:
            sql = "select id,title,abstract,body " \
                  "from dg_crawler.news " \
                  "where language_id in (1866)" \
                  f"limit {self.current_pointer},{self.batch_size};"
            # "where language_id in (1748,1779,1813,1814,1866,1867,1930,1952,1926,1982,2275,2005,2036,1797,2065,2208,2207,2242,2227,2238)" \

            self.dg_cur.execute(sql)
            self.current_pointer+=self.batch_size
            data = self.dg_cur.fetchall()
            ids = []
            text_set = []
            # print(data[0])
            for item in data:
                a,b,c='','',''
                if item[1]!=None:
                    a = item[1]
                if item[2]!=None:
                    b = item[2]
                if item[3]!=None:
                    c = item[3]
                ids.append(item[0])
                text_set.append(''.join([a,b,c]))
            return ids,text_set
        else:
            return None

    def insert_annotations(self,data):
        for item in data:
            sql = f"insert into news_annotation.news_annotation (news_id,about_china) values ({item[0]},{item[1]});"
            self.annotation_cur.execute(sql)
        self.annotation_connet.commit()


if __name__ == '__main__':
    database_handler = database_handler()
    ids,text_set = database_handler.get_data()

    classifier = classifier()
    labels = classifier.classify(text_set)

    database_handler.insert_annotations(zip(ids,labels))