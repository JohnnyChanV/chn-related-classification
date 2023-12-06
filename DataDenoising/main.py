import os

from tqdm import tqdm
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()
from Dataset import Dataset
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# SVM、线性SVM
from sklearn.svm import SVC, LinearSVC
# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB, MultinomialNB
# 决策树
from sklearn.tree import DecisionTreeClassifier
# 随机森林、GBDT
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained('xlm-roberta-base')
def tokenizer(text):
    text = text.replace('\n', '').replace('\t', '')
    for p in string.punctuation:
        text = text.replace(p, ' ' + p + ' ')
    return tk.tokenize(text, truncation=True)

def train(vectorizer,train_data,model):
    train_sequences = vectorizer.transform([i[1] for i in train_data])
    train_labels = [i[-1] for i in train_data]
    model.fit(train_sequences,train_labels)
    return model

def test(vectorizer,test_data,model):
    test_texts = [i[1] for i in test_data]
    test_sequences = vectorizer.transform(test_texts)
    true_labels = [i[-1] for i in test_data]
    indexes = [i[0] for i in test_data]
    pred_labels = model.predict(test_sequences)
    return list(zip(indexes,test_texts,true_labels,pred_labels))

def get_confirm_instances(outputs):
    model_num = len(outputs)
    label_num = len(outputs[0])
    confirm_instances = []
    next_test = []
    for label_index in range(label_num):
        is_confirm = True
        for model_index in range(model_num-1):
            if outputs[model_index][label_index][-1] != outputs[model_index+1][label_index][-1]:
                is_confirm = False
        if is_confirm:
            confirm_instances.append(outputs[0][label_index])
        else:
            next_test.append(outputs[0][label_index])
    return confirm_instances,next_test

def distinct(data):
    distinct_data = []
    indexes = list(set([i[0] for i in data]))
    for index in indexes:
        for instance in data:
            if index == instance[0]:
                distinct_data.append(instance)
    return distinct_data


def save_confirm_instances(instances,name):
    data = []
    for index,item in enumerate(instances):
        data.append({
            'index':index,
            'vi_title':'',
            'vi_body':item[1],
            'label':bool(item[-1])
        })
    json.dump(data,open(f'data/enfiltered_{name}','w'),ensure_ascii=False)

if __name__ == "__main__":
    # save_confirm_instances([(1,'',True)],'data.json')
    args = json.load(open('DataDenoising/config.json'))
    vectorizer = TfidfVectorizer(
        lowercase=True,  # 是否将单词转为小写（如果为True的话就可以缩小特征空间）
        analyzer="word",  # 词级别分析（将文档在单词级别转成向量）
        tokenizer=tokenizer,  # 分词器，一个函数，给定一个文本返回词列表。
        stop_words=None,  # 停用词表， 包含停用词的一个列表。
        max_features=args['max_vocab'],  # 最大词数目，如果给定了最大词数目，则只会保留出现频率最高的max_features个词
    )

    for train_json in os.listdir('data'):
        if train_json.startswith('cnAnnotated') and train_json.endswith('json'):
            train_dataset = Dataset(path=f"data/{train_json}")
            train_data = train_dataset.get_data()

            test_dataset = Dataset(path=f"data/{train_json.replace('cnAnnotated','enAnnotated')}")
            test_data = test_dataset.get_data()
            print(f"[info]: filtering {train_json}")
        else:
            continue

        vectorizer.fit([i[1] for i in train_data])

        for epoch in range(args['epochs']):
            models = [MultinomialNB(), RandomForestClassifier(), GradientBoostingClassifier()]
            models_outputs = []
            print(f"---------------------------------------------------------------")
            print(f"[INFO]: SIZE OF TRAIN SET IN EPOCH {epoch}: {len(train_data)}.")
            print(f"[INFO]: SIZE OF TEST SET IN EPOCH {epoch}: {len(test_data)}.")
            for clf in models:
                #训练三个模型 并取三个模型的测试集输出
                print(f"[INFO]:Epoch:{epoch}, Model:{clf.__str__()}")
                model = train(vectorizer,train_data,clf)
                test_outputs = test(vectorizer,test_data,model)
                test_indexes = [i[0] for i in test_outputs]
                test_texts = [i[1] for i in test_outputs]
                test_true_labels = [i[2] for i in test_outputs]
                test_pred_labels = [i[3] for i in test_outputs]
                print(classification_report(test_true_labels,test_pred_labels,digits=4))
                models_outputs.append(list(zip(test_indexes,test_texts,test_pred_labels)))

            confirm_instances,next_test = get_confirm_instances(models_outputs)
            print(f"[info]: size of confirmed set {len(confirm_instances)}")
            # print(confirm_instances)
            save_confirm_instances(confirm_instances,train_json.split('en2')[-1])
            print(f"[info]: {train_json.split('en2')[-1].replace('.json','')} Finished!")
    print("----------------------------------------------------------------------------------")














