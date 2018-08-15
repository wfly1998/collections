import urllib
import urllib.request
from urllib.request import urlopen
from urllib.error import URLError
import sys
import re
import json


urllib.request.socket.setdefaulttimeout(30)
reg_text = re.compile(r'(?<=TRANSLATED_TEXT=).*?;')

class Translater(object):
	
	'''
	Translate word from youdao.com
	'''
	@staticmethod
	def translateWord(word, retryTimes=10):
		d = {}
		searchUrl = "http://dict.youdao.com/search?q=" + word + "&keyfrom=dict.index"
		retry = 0
		while retry < retryTimes:
			try:
				response = urlopen(searchUrl).read().decode("utf-8")
				break
			except:
				retry += 1
				if retry >= retryTimes:
					return {}
		index1 = response.find(r'<div class="baav">')
		index2 = response.find(r'</div>', index1)
		baav = response[index1: index2]
		words = baav.split(r'<span class="pronounce">')
		for word in words:
			index1 = word.find(r'<span class="phonetic">')
			if index1 == -1:
				continue
			length = len(r'<span class="phonetic">')
			index2 = word.find(r'</span>', index1)
			pho = word[index1+length: index2]
			if d.get('EnPron', 'En') == 'En':
				d['EnPron'] = pho
			else:
				d['AmPron'] = pho
			pass

		searchSuccess = re.search(r'(?s)<div class="trans-container">.*?<ul>.*?</div>', response)

		if searchSuccess:
			means = re.findall(r'(?m)<li>(.*?)</li>', searchSuccess.group())
			d['means'] = list(means)
			return d
		else:
			return {}
	
	'''
	Translate sentense from Google Translate
	'''
	@staticmethod
	def translateSentense(text, f='en', t='zh-cn'):
		url_google = 'http://translate.google.cn'
		user_agent = r'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) ' \
		r'Chrome/44.0.2403.157 Safari/537.36'
		
		values = {'hl': 'zh-cn', 'ie': 'utf-8', 'text': text, 'langpair': '%s|%s' % (f, t)}
		
		value = urllib.parse.urlencode(values)
		req = urllib.request.Request(url_google + '?' + value)
		req.add_header('User-Agent', user_agent)
		try:
			response = urllib.request.urlopen(req)
		except URLError:
			return 'Translate failed.'
		content = response.read().decode('utf-8')
		data = reg_text.search(content)
		result = data.group(0).strip(';').strip('\'')
		return result

