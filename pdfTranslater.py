from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed

from translater import Translater

import logging


if __name__ == '__main__':
	
	logging.propagate = False
	logging.getLogger().setLevel(logging.ERROR)

	file = '3DMM_Edge.pdf'
	output = '3dmm_edge.txt'
	with open(file, 'rb') as fp:
		praser = PDFParser(fp)
		doc = PDFDocument()
		praser.set_document(doc)
		doc.set_parser(praser)
		doc.initialize()
		if not doc.is_extractable:
			raise PDFTextExtractionNotAllowed
			exit(0)
		rsrcmgr = PDFResourceManager()
		laparams = LAParams()
		device = PDFPageAggregator(rsrcmgr, laparams=laparams)
		
		interpreter = PDFPageInterpreter(rsrcmgr, device)

		with open(output, 'wt') as f:
			p = 0
			for page in doc.get_pages():
				p += 1
				print('Page %d is translating.' % p)
				interpreter.process_page(page)
				layout = device.get_result()
				lay = 0
				for x in layout:
					if (isinstance(x, LTTextBoxHorizontal)):
						lay += 1
						print('Sentense %d is translating.' % lay)
						result = x.get_text().replace('-\n', '').replace('\n', ' ')
						trans = Translater.translateSentense(result)
						f.write(result + '\n'*2 + trans + '\n'*4)
pass


