import copy
from pprint import pprint

from sklearn.metrics import classification_report, f1_score, accuracy_score

from Dataset import *
from argparse import ArgumentParser
import os
from model import *

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["http_proxy"] = "http://192.168.235.34:7890"
os.environ["https_proxy"] = "http://192.168.235.34:7890"

import warnings

warnings.filterwarnings("ignore")

# 创建解析器对象
parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--LLM", type=str, default='xlm-roberta-base')
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--class_num", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--early_stop", type=int, default=2)
parser.add_argument("--valid_round", type=int, default=2)
parser.add_argument("--data_path", type=str, default='data/enfiltered_id_data.json')
parser.add_argument("--test_data_path", type=str, default='data/cnfiltered_en_data.json')
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--embed_dim", type=int, default=300)
parser.add_argument("--num_worker", type=int, default=4)
parser.add_argument("--use_llm", action='store_true')
parser.add_argument("--x_lingual", action='store_true')

args = parser.parse_args()


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []
    print('[INFO]: Training..')
    loader_ = tqdm(loader)
    for index, data in enumerate(loader_):
        indexes, input_ids, mask, labels = data
        optimizer.zero_grad()
        logits = model(input_ids.to(device), mask.to(device))
        proba = torch.softmax(logits, -1)

        loss = criterion(logits.view(-1, 2), labels.view(-1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_true += labels.flatten().tolist()
        y_pred += proba.argmax(dim=-1).flatten().tolist()
        ##show info
        loader_.set_postfix(loss=loss.item())
    # report = classification_report(y_true, y_pred)
    # print(report)
    return total_loss / len(loader)


def valid(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        print('[INFO]: Validating..')
        loader_ = tqdm(loader)
        for index, data in enumerate(loader_):
            indexes, input_ids, mask, labels = data
            logits = model(input_ids.to(device), mask.to(device))
            proba = torch.softmax(logits, -1)

            loss = criterion(logits.view(-1, 2), labels.view(-1).to(device))
            total_loss += loss.item()

            y_true += labels.flatten().tolist()
            y_pred += proba.argmax(dim=-1).flatten().tolist()
            ##show info
            loader_.set_postfix(loss=loss.item())
            report = classification_report(y_true, y_pred)
            acc_no_zero = accuracy_score(y_true, y_pred)
            f1_score_ = f1_score(y_true, y_pred, average='macro')
        print('Accuracy:', acc_no_zero)
        print(report)
        return total_loss / len(loader), acc_no_zero, f1_score_


def test(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        print('[INFO]: Testing..')
        for index, data in enumerate(tqdm(loader)):
            indexes, input_ids, mask, labels = data
            logits = model(input_ids.to(device), mask.to(device))
            proba = torch.softmax(logits, -1)
            y_true += labels.flatten().tolist()
            y_pred += proba.argmax(dim=-1).flatten().tolist()

            report = classification_report(y_true, y_pred, digits=4)
            acc_no_zero = accuracy_score(y_true, y_pred)
        print('Accuracy:', acc_no_zero)
        print(report)
        open('results.txt','a').write(f"{item['path']}")
        open('results.txt','a').write(report)
        open('results.txt','a').write(f"\nAccuracy: {acc_no_zero}\n\n")
        return


if __name__ == '__main__':
    pprint(args)
    torch.manual_seed(args.seed)
    open('results.txt','w').write('')

    if not args.x_lingual:
        train_data = RawDataset(args.data_path).get_data()
        test_data = RawDataset(args.test_data_path).get_data()

        v, t = train_test_split(test_data, test_size=0.33, random_state=42)
        train_loader = data_loader(train_data, args)
        test_loader = data_loader(t, args)
        valid_loader = data_loader(v, args)
    else:
        args.data_path = args.data_path.split('/')[0]
        x_lingual_train_data = []
        x_lingual_valid_data = []
        x_lingual_test_data = []

        mono_lingual_dataloader = []
        for json in os.listdir(args.data_path):
            # if json.startswith('enfiltered') and json.endswith('.json'):
            #     tr = RawDataset(f"{args.data_path}/{json}").get_data()
            #     x_lingual_train_data += tr
            #
            # if json.startswith('cnfiltered') and json.endswith('.json'):
            #     v = RawDataset(f"{args.data_path}/{json}").get_data()
            #     x_lingual_valid_data += v
            #     mono_lingual_dataloader.append({'loader': data_loader(v, args, shuffle=False), "path": json})

            # if json.startswith('cnfiltered') and json.endswith('.json'):
            #     tr,v= RawDataset(f"{args.data_path}/{json}").get_train_test_data()
            #     x_lingual_train_data += tr
            #     x_lingual_valid_data += v
            #     mono_lingual_dataloader.append({'loader':data_loader(v, args, shuffle=False),"path":json})

            if 'filtered' in json and json.endswith('.json'):
                tr, v = RawDataset(f"{args.data_path}/{json}").get_train_test_data()
                x_lingual_train_data += tr
                v, t = train_test_split(v, test_size=0.33, random_state=42)
                x_lingual_valid_data += v
                # x_lingual_test_data += t
                mono_lingual_dataloader.append({'loader': data_loader(t, args, shuffle=False), "path": json})

        print("[info]: Processing TrainLoader...")
        train_loader = data_loader(x_lingual_train_data, args)
        print("[info]: Processing ValidationLoader...")
        valid_loader = data_loader(x_lingual_valid_data, args, shuffle=False)
        # test_loader = data_loader(x_lingual_test_data, args, shuffle=False)

    model = Classifier(args)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    device = torch.device(args.device)
    print(f"[info]: Current device {device}")
    model.to(device)
    bestloss = 1e9
    bestacc = -1e9
    bestf1 = -1e9
    notbetter_count = 0
    bestmodel = model
    for epoch in range(args.epochs):
        if notbetter_count > args.early_stop:
            print("[INFO]: Not better, Early stop.")
            break
        print(f"---------Epoch:{epoch}---------")
        train(model, train_loader, optimizer, criterion, device)
        if epoch % args.valid_round == 0:
            notbetter_count += 1
            thisloss, thisacc, thisf1 = valid(model, valid_loader, criterion, device)
            if thisf1 > bestf1:
                bestf1 = thisf1
                notbetter_count = 0
                torch.save(model, f"best_model.pt")
                bestmodel = model
                print(f"Best Model Saved! With Acc:{thisacc}, AvgLoss:{thisloss}, Macro F1:{thisf1}")
    model = torch.load(f"best_model.pt")
    # model = bestmodel

    try:
        print(f"[INFO]：X-lingual Test:")
        test(model, test_loader, device)
    except:
        pass

    print(f"[INFO]:Monolingual Test")

    for item in mono_lingual_dataloader:
        print(f"\t[INFO]: Language {item['path']}")
        test(model, item['loader'], device)
    pprint(args)
