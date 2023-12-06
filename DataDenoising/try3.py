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

from sklearn.model_selection import cross_val_score

def tokenizer(text):
    text = text.replace('\n', '').replace('\t', '')
    for p in string.punctuation:
        text = text.replace(p, ' ' + p + ' ')
    return text.split()

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


if __name__ == "__main__":
    args = json.load(open('config.json'))
    vectorizer = TfidfVectorizer(
        lowercase=True,  # 是否将单词转为小写（如果为True的话就可以缩小特征空间）
        analyzer="word",  # 词级别分析（将文档在单词级别转成向量）
        tokenizer=tokenizer,  # 分词器，一个函数，给定一个文本返回词列表。
        stop_words=None,  # 停用词表， 包含停用词的一个列表。
        max_features=args['max_vocab'],  # 最大词数目，如果给定了最大词数目，则只会保留出现频率最高的max_features个词
    )

    A_dataset = Dataset(path="data/enAnnotated_en2vi_data.json") #1.5w
    A_data = A_dataset.get_data()
    B_dataset = Dataset(path="data/cnAnnotated_en2vi_data.json") #5.5w
    B_data = B_dataset.get_data()


#Stage 1:----------------------------------------------------
    vectorizer.fit([i[1] for i in A_data])
    models = [MultinomialNB(), RandomForestClassifier(), GradientBoostingClassifier()]
    models_outputs = []
    print(f"--------------STAGE 1-----------------------------------")
    print(f"[INFO]: SIZE OF TRAIN SET: {len(A_data)}.")
    print(f"[INFO]: SIZE OF TEST SET: {len(B_data)}.")
    for clf in models:
        #训练三个模型 并取三个模型的测试集输出
        print(f"[INFO]: Model:{clf.__str__()}")
        model = train(vectorizer, A_data, clf)
        test_outputs = test(vectorizer, B_data, model)
        test_indexes = [i[0] for i in test_outputs]
        test_texts = [i[1] for i in test_outputs]
        test_true_labels = [i[2] for i in test_outputs]
        test_pred_labels = [i[3] for i in test_outputs]
        print(classification_report(test_true_labels,test_pred_labels,digits=4))
        models_outputs.append(list(zip(test_indexes,test_texts,test_pred_labels)))

    cnAnnotated_confirm_instances,not_confirm_instances = get_confirm_instances(models_outputs)
    cnAnnotated_confirm_instances = distinct(cnAnnotated_confirm_instances)

#Stage 2:----------------------------------------------------
    vectorizer.fit([i[1] for i in cnAnnotated_confirm_instances])
    models = [MultinomialNB(), RandomForestClassifier(), GradientBoostingClassifier()]
    models_outputs = []
    print(f"------------------STAGE 2------------------------------")
    print(f"[INFO]: SIZE OF TRAIN SET: {len(cnAnnotated_confirm_instances)}.")
    print(f"[INFO]: SIZE OF TEST SET: {len(A_data)}.")
    for clf in models:
        #训练三个模型 并取三个模型的测试集输出
        print(f"[INFO]: Model:{clf.__str__()}")
        model = train(vectorizer,cnAnnotated_confirm_instances,clf)
        test_outputs = test(vectorizer, A_data, model)
        test_indexes = [i[0] for i in test_outputs]
        test_texts = [i[1] for i in test_outputs]
        test_true_labels = [i[2] for i in test_outputs]
        test_pred_labels = [i[3] for i in test_outputs]
        print(classification_report(test_true_labels,test_pred_labels,digits=4))
        models_outputs.append(list(zip(test_indexes,test_texts,test_pred_labels)))

    enAnnotated_confirm_instances,not_confirm_instances = get_confirm_instances(models_outputs)
    enAnnotated_confirm_instances = distinct(enAnnotated_confirm_instances)

#Stage 3:----------------------------------------------------
    merge_confirm_instances = cnAnnotated_confirm_instances+enAnnotated_confirm_instances
    vectorizer.fit([i[1] for i in merge_confirm_instances])
    models = [MultinomialNB(), RandomForestClassifier(), GradientBoostingClassifier()]
    models_outputs = []
    print(f"------------------STAGE 3------------------------------")
    print(f"[INFO]: SIZE OF TRAIN SET: {len(merge_confirm_instances)}.")
    print(f"[INFO]: SIZE OF TEST SET: {len(A_data)}.")
    for clf in models:
        # 训练三个模型 并取三个模型的测试集输出
        print(f"[INFO]: Model:{clf.__str__()}")
        model = train(vectorizer, cnAnnotated_confirm_instances, clf)
        test_outputs = test(vectorizer, A_data, model)
        test_indexes = [i[0] for i in test_outputs]
        test_texts = [i[1] for i in test_outputs]
        test_true_labels = [i[2] for i in test_outputs]
        test_pred_labels = [i[3] for i in test_outputs]
        print(classification_report(test_true_labels, test_pred_labels, digits=4))
        scores = cross_val_score(clf, vectorizer.transform([i[1] for i in A_data]),
                                 [i[-1] for i in A_data], cv=10)
        print(f"[Result]: Cross Validation Mean F1:{scores.mean()}")
