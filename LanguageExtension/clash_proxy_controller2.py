import requests
from pprint import pprint
import json

class clash_proxy_controller(): #实现了用多节点打游击战进行爬取的方法！
    def __init__(self):
        self.control_url = "http://127.0.0.1:9090/proxies/Proxy"
        res = requests.get(self.control_url)
        self.proxy_list = []
        for p in json.loads(res.text)['all']:
            if '美国' in p:
                self.proxy_list.append(p)

        requests.put(self.control_url,json={'name':self.proxy_list[0]})
        print(self.proxy_list)
        self.current_proxy = 0

    def change_proxy(self):
        self.current_proxy+=1
        proxy = (self.current_proxy+1) % len(self.proxy_list)
        requests.put(self.control_url,json={'name':self.proxy_list[proxy]})
        print(f"[info]: 节点已更换为「{self.proxy_list[proxy]}」")
