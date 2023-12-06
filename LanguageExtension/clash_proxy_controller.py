import requests
from pprint import pprint
import json
import os

class clash_proxy_controller(): #实现了用多节点打游击战进行爬取的方法！
    def __init__(self):
        os.environ['https_proxy'] = ''
        os.environ['http_proxy'] = ''
        self.control_url = "http://127.0.0.1:9090/proxies/GLOBAL"
        res = requests.get(self.control_url)
        print(res)
        self.proxy_list = []
        print(json.loads(res.text)['all'])

        for p in json.loads(res.text)['all']:
            if '香港' in p:
                self.proxy_list.append(p)
        self.proxy_list = self.proxy_list
        pprint(self.proxy_list)
        requests.put(self.control_url,json={'name':self.proxy_list[0]})
        self.current_proxy = 0

    def change_proxy(self):
        os.environ['https_proxy'] = ''
        os.environ['http_proxy'] = ''
        self.current_proxy+=1
        proxy = (self.current_proxy) % len(self.proxy_list)
        requests.put(self.control_url,json={'name':self.proxy_list[proxy]})
        print(f"[info]: 节点已更换为「{self.proxy_list[proxy]}」")
