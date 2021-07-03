# -*- coding: utf-8 -*-

import json
import requests

class TPLogin:
    headers = None
    logined = False
    def __init__(self, url='192.168.1.1', pwd=None):
        self.url = 'http://' + url + '/'
        if pwd:
            self.login(pwd)

    def _request(self, post: map, with_stok=True) -> map:
        url = self.url
        if with_stok:
            url += 'stok=' + self._stok
        return json.loads(requests.post(url, json.dumps(post)).text)

    def login(self, pwd: str) -> bool:
        password = TPLogin._orgAuthPwd(pwd)
        res = self._request({'method': 'do', 'login': {'password': password}}, with_stok=False)
        if res['error_code'] == 0:
            self._stok = res['stok']
            self.logined = True
        return self.logined

    def reboot(self):
        assert self.logined
        res = self._request({'system': {'reboot': None}, 'method': 'do'})
        if res['error_code'] == 0:
            return res['wait_time']
        return -1

    @staticmethod
    def _orgAuthPwd(a: str) -> str:
        return TPLogin._securityEncode(
                "RDpbLfCPsJZ7fiv", 
                a, 
                "yLwVl0zKqws7LgKPRQ84Mdt708T1qQ3Ha7xv3H7NyU84p21BriUWBU43odz3iP4rBL3cD02KZciXTysVXiV8ngg6vL48rPJyAUw0HurW20xqxv9aYb4M9wK1Ae0wlro510qXeU07kV57fQMc8L6aLgMLwygtc0F10a0Dg70TOoouyFhdysuRMO51yY5ZlOZZLEal1h0t9YQW0Ko7oBwmCAHoic4HYbUyVeU3sfQ1xtXcPcf1aT303wAQhv66qzW"
            )

    @staticmethod
    def _securityEncode(a: str, b: str, c: str) -> str:
        e = ''
        g = len(a)
        h = len(b)
        k = len(c)
        l = 187
        n = 187
        f = max(g, h)
        for p in range(f):
            n = 187
            l = 187
            if p >= g:
                n = ord(b[p])
            elif p >= h:
                l = ord(a[p])
            else:
                l = ord(a[p])
                n = ord(b[p])
            e += c[(l^n)%k]
        return e

def main():
    tp = TPLogin('192.168.1.1', 'password')
    time = tp.reboot()
    if time == -1:
        print('reboot failed')
        exit(1)
    print('reboot succeed, please wait for {} seconds'.format(time))
    exit(0)

if __name__ == '__main__':
    main()

