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
    test_sequences = vectorizer.transform([i[1] for i in test_data])
    true_labels = [i[-1] for i in test_data]
    pred_labels = model.predict(test_sequences)
    return true_labels,pred_labels


if __name__ == "__main__":
    args = json.load(open('config.json'))
    vectorizer = TfidfVectorizer(
        lowercase=True,  # 是否将单词转为小写（如果为True的话就可以缩小特征空间）
        analyzer="word",  # 词级别分析（将文档在单词级别转成向量）
        tokenizer=tokenizer,  # 分词器，一个函数，给定一个文本返回词列表。
        stop_words=None,  # 停用词表， 包含停用词的一个列表。
        max_features=args['max_vocab'],  # 最大词数目，如果给定了最大词数目，则只会保留出现频率最高的max_features个词
    )

    dataset = Dataset(path=args['path'])
    train_data,test_data = dataset.get_train_test_data()

    vectorizer.fit([i[1] for i in train_data])

    models = [MultinomialNB(),RandomForestClassifier(),GradientBoostingClassifier()]
    for clf in models:
        print(f"[INFO]: Model:{clf.__str__()}")
        model = train(vectorizer,train_data,clf)
        true_labels,pred_labels = test(vectorizer,test_data,model)
        print(classification_report(true_labels,pred_labels,digits=4))