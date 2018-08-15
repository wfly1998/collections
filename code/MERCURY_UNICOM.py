# -*- coding: utf-8 -*-

import urllib
from urllib import request, parse
from http import cookiejar
import json
import re

class obj(object):
    pass

SYN,ASYN=tuple(range(2))
TDDP_INSTRUCT,TDDP_WRITE,TDDP_READ,TDDP_UPLOAD,TDDP_DOWNLOAD,TDDP_RESET,TDDP_REBOOT,TDDP_AUTH,TDDP_GETPEERMAC,TDDP_CONFIG,TDDP_CHGPWD,TDDP_LOGOUT=tuple(range(12))
PARSE_INIT,PARSE_NOTE,PARSE_CMD,PARSE_ID,PARSE_INDEX,PARSE_VALUE,PARSE_ERR=tuple(range(7))
ENONE = 0
EUNAUTH = 7
authInfo = [None, '', '', '', '']
class MELOGIN(object):
    
    domainUrl = 'http://melogin.cn/'
    
    def __init__(self):
        self._result = obj()
        self._initResult()
        self._pagePRHandle = None
        self._orgAuthPwd = lambda a: self._securityEncode(a, "RDpbLfCPsJZ7fiv", "yLwVl0zKqws7LgKPRQ84Mdt708T1qQ3Ha7xv3H7NyU84p21BriUWBU43odz3iP4rBL3cD02KZciXTysVXiV8ngg6vL48rPJyAUw0HurW20xqxv9aYb4M9wK1Ae0wlro510qXeU07kV57fQMc8L6aLgMLwygtc0F10a0Dg70TOoouyFhdysuRMO51yY5ZlOZZLEal1h0t9YQW0Ko7oBwmCAHoic4HYbUyVeU3sfQ1xtXcPcf1aT303wAQhv66qzW")
        self._encodePara = lambda a: parse.quote(a)
        
        self.linkUp = lambda: self._inStr('wan -linkUp')
        self.linkDown = lambda: self._inStr('wan -linkDown')
    
    def _securityEncode(self, a, b, c):
        charCodeAt = lambda s, p: int(s.encode()[p])
        d = ''
        f = len(a)
        h = len(b)
        m = len(c)
        e = f if f > h else h
        g = 0
        while g < e:
            l = k = 187
            if g >= f:
                l = charCodeAt(b, g)
            elif g >= h:
                k = charCodeAt(a, g)
            else:
                k = charCodeAt(a, g)
                l = charCodeAt(b, g)
            d += c[(k^l)%m]
            g += 1
        return d
        
    def _auth(self, a=None):
        b = a
        c = self.domainUrl + '?code=' + str(TDDP_AUTH) + '&asyn=0'
        if a is None or len(b) == 0:
            self._result.errorno = EUNAUTH
            return (self._result.errorno, self._result)
        a = ''
        self._initResult()
        self._session = self._securityEncode(authInfo[3], b, authInfo[4])
        c += '&id=' + self._encodePara(self._session)
        self._request(c, a.encode())
        self._parseAuthRlt()
        return self._result
        
    
    def _request(self, url, data=None, func=None):
        try:
            resp = request.urlopen(url, data)
        except urllib.error.HTTPError as e:
            resp = e
        t = resp.read().decode()
        a = t.split()
        self._result.errorno = int(a[0])
        self._result.data = '\n'.join(a[1:])
        if not data is None:
            self._parseAuthRlt()
        if not func is None:
            func(self._result)
    
    def _initResult(self):
        self._result.errorno = 0
        self._result.data = ""
        self._result.timeout = 1
        
    def _parseAuthRlt(self):
        a = self._result.data
        a = a.split()
        if self._result.errorno == EUNAUTH:
            authInfo[1] = a[0]
            authInfo[2] = a[1]
            authInfo[3] = a[2]
            authInfo[4] = a[3]
            # ???
        return a
        
    def rsf(self, pwd):
        b = self.domainUrl + '?code=' + str(TDDP_READ) + '&asyn=1'
        self._request(b, data='\n'.encode())
        if self._result.errorno == EUNAUTH:
            self._parseAuthRlt()
        result = self._auth(self._orgAuthPwd(pwd))
        return result.errorno == ENONE
        
    def _orgUrl(self, a):
        sessionKey = 'id'
        a += ('?' if a.find('?') == -1 else '&') + sessionKey + '=' + self._encodePara(self._session)
        return a
    
    def _inStr(self, a):
        c = self.domainUrl + '?code=' + str(TDDP_INSTRUCT) + '&asyn=0'
        self._initResult()
        self._request(self._orgUrl(c), a.encode())
        return self._result

class CU_91WiFi(object):
    '''
    Only used for UJN
    '''
    def __init__(self, un, pwd):
        self.url_api = 'http://139.198.3.98/mgr/api/'
        self.url_ujn = 'http://139.198.3.98/sdjd/'
        self.un = str(un)
        self.pwd = str(pwd)
        cj = cookiejar.CookieJar()
        self.opener = request.build_opener(request.HTTPCookieProcessor(cj))
        self.opener.addheaders = [('Connection', 'keep-alive'), 
                                  ('Accept-Language', 'zh-cn'), 
                                  ('User-Agent', '91wifi/17 CFNetwork/811.5.4 Darwin/16.7.0'), 
                                  ('Accept-Encoding', 'deflate'), 
                                  ('Connection', 'keep-alive')]
        self.urldecode = lambda q: [tuple(s.split('=')) for s in q.split('&')]
    
    def _httpResp(self, url, data=None):
        d = data
        if d:
            d = data.encode()
        return self.opener.open(url, data=d)
    
    def _httpRead(self, url, data=None):
        return self._httpResp(url, data).read()
    
    def _isNeedPwd(self):
        '''
        cannot understand this
        '''
        url = self.url_api + 'auth/pwd.html'
        data = 'user_name=%s&school_id=510592' % self.un
        recv = self._httpRead(url+'?'+data, data=data)
        j = json.loads(recv)
        return j['status'] if isinstance(j['status'], int) else -1
    
    def _checkVersion(self, version):
        '''
        used for iOS, no use
        '''
        url = self.url_api + 'ios/upgrade.html'
        data = 'version=%d' % int(version)
        recv = self._httpRead(url, data=data)
        j = json.loads(recv)
        return j['status'] if isinstance(j['status'], int) else -1
    
    def _getLocation(self):
        '''
        get location of 302
        '''
        url = 'http://www.126.com/'
        resp = self._httpResp(url)
        headers = resp.getheaders()
        loc = ''
        for k, v in headers:
            if k == 'Location':
                loc = v
        if loc == '':
            urls = re.findall(r'window.location.href="(.*?)"', resp.read().decode())
            if len(urls) == 1:
                loc = urls[0]
        return loc
    
    def _getIP(self, loc):
        '''
        get wlanuserip and basip from url
        '''
        p = parse.urlparse(loc)
        d = self.urldecode(p.query)
        w, b = '', ''
        for k, v in d:
            if k == 'wlanuserip':
                w = v
            elif k == 'basip':
                b = v
        self.ip = (w, b)
     
    def _reLoad(self):
        '''
        set the Cookie of 'JSESSIONID'
        '''
        url = self.url_ujn + '?wlanuserip=%s&wlanacname=SDJN-XZ-ME60-B2&basip=%s' % self.ip
        self._httpRead(url)
    
    def _login(self):
        '''
        inner login method
        '''
        url = self.url_ujn + '/services/portal/portalAuth'
        data = 'lpsUserName=%s&lpsPwd=%s&wlanuserip=%s&basip=%s&school_id=510592' % (self.un, self.pwd, *self.ip)
        recv = self._httpRead(url+'?'+data, data=data).decode()
        j = json.loads(recv)
        if isinstance(j['msg'], str):
            print(j['msg'])
        return j['success'] if isinstance(j['success'], bool) else False
    
    def login(self):
        '''
        login
        '''
        self._getIP(self._getLocation())
        self._reLoad()
        return self._login()
    
    def logout(self):
        '''
        logout
        '''
        url = self.url_ujn + '/services/portal/portalExit'
        data = 'lpsUserName=%s&lpsPwd=%s&wlanuserip=%s&basip=%s' % (self.un, self.pwd, *self.ip)
        recv = self._httpRead(url+'?'+data, data=data)
        j = json.loads(recv)
        return j['success'] if isinstance(j['success'], bool) else False


if __name__ == '__main__':
    
    s = True
    
    '''
    m = MELOGIN()
    s = m.rsf('mercuryPWD')
    print(s)
    '''
    
    if s:
        m.linkDown()
        m.linkUp()
        c = CU_91WiFi('UnicomUserName', 'UnicomPWD')
        print(c.login())
    