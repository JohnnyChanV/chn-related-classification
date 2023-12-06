import time

from tqdm import tqdm
import pandas as pd
from googletranslatepy import Translator
import json
import concurrent.futures
import os
from clash_proxy_controller import clash_proxy_controller


def translate_text(translator, text):
    tt = translator.translate(text)
    return tt


def process_item(translator, item):
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    i = {
        'index': None,
        'vi_title': None,
        'vi_body': None,
        'label': None
    }
    i['index'] = item[3]
    i['vi_title'] = translate_text(translator, item[0])
    i['vi_body'] = translate_text(translator, item[1])
    i['label'] = item[2]
    return i


def get_pre_list_index(path):
    indexes = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            indexes.append(eval(line)['index'])
    indexes = list(set(indexes))
    return indexes


if __name__ == "__main__":
    clash_proxy_controller = clash_proxy_controller()
    for target in ['de','ru','it','pl','el','ja','ko','es','pt','fr']:
        translator = Translator(source='en', target=target)

        for annotated_lan in ['cn','en']:
            save_path = f'data/{annotated_lan}processed_' + target + ".json"
            print(f'[info]: loading..{save_path}')
            raw = json.load(open(f'data/{annotated_lan}Annotated_en2en_data.json', 'r', encoding='utf-8'))
            temp = list(zip(list([i['vi_title'] for i in raw]), list([i['vi_body'] for i in raw]),
                                list([i['label'] for i in raw]), list([i['index'] for i in raw])))
            raw_list = []
            for item in temp:
                if item[0] != False and item[1] != False:
                    raw_list.append(item)

            try:
                pre_indexes = get_pre_list_index(save_path)
            except:
                pre_indexes = []


            while len(raw_list) > len(pre_indexes):
                try:
                    pre_indexes = get_pre_list_index(save_path)
                    print(f"[info]: current Size:{len(pre_indexes)}")
                except:
                    pre_indexes = []

                num_items = len(raw_list)
                batch_size = 64
                num_batches = num_items // batch_size + 1

                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    for batch_num in (range(num_batches)):
                        start_index = batch_num * batch_size
                        end_index = min(start_index + batch_size, num_items)
                        batch = raw_list[start_index:end_index]

                        futures = []
                        running_num = 0
                        print(
                            f"[INFO]: Processing batch {batch_num + 1}/{num_batches}...Annotated:{annotated_lan}, target:{target}")
                        for item in batch:
                            # if item[3] not in pre_indexes and item[0] != False and item[1] != False:
                            future = executor.submit(process_item, translator, item)
                            futures.append(future)
                            running_num += 1

                        print(f"[INFO]: size:{running_num}")
                        for future in (concurrent.futures.as_completed(futures)):
                            result = future.result()
                            if result['vi_title'] != False:
                                open(save_path, 'a', encoding='utf-8').write(str(result) + '\n')
                        clash_proxy_controller.change_proxy()
                        # if running_num > 20:
                        #     print("[info]: stop for defence.")
                        #     time.sleep(2)

            with open(save_path, 'r', encoding='utf-8') as f:
                datas = []
                data_indexes = []
                lens = []
                for item in f.readlines():
                    i = eval(item)
                    if i['index'] in data_indexes:
                        continue
                    datas.append(i)
                    data_indexes.append(i['index'])
                    if i['vi_body'] != False:
                        lens.append(len(i['vi_body'].split(' ')))
                print(f'[info]: {annotated_lan}Annotated_en2{target}_data.json Finished! Size:{len(datas)}')
                json.dump(datas, open(f'{annotated_lan}Annotated_en2{target}_data.json', 'w', encoding='utf-8'),
                          ensure_ascii=False)
