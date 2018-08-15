from translater import Translater
import xlwt

if __name__ == '__main__':
	file = 'wordsC.txt'
	
	with open(file, 'rt') as f:
		book = xlwt.Workbook(encoding='utf-8')
		sheet = book.add_sheet('Sheet1')
		
		style = xlwt.XFStyle()
		alignment = xlwt.Alignment()
		alignment.vert = xlwt.Alignment.VERT_TOP
		alignment.wrap = xlwt.Alignment.WRAP_AT_RIGHT
		borders = xlwt.Borders()
		borders.left = xlwt.Borders.THIN
		borders.right = xlwt.Borders.THIN
		borders.top = xlwt.Borders.THIN
		borders.bottom = xlwt.Borders.THIN
		
		style.alignment = alignment
		style.borders = borders
		
		row = 0
		for line in f:
			words = line.split()
			col = 0
			for word in words:
				sheet.col(col).width = 256 * 30
				print('Translating:', word)
				d = Translater.translateWord(word)
				sheet
				sheet.write(row, col, word+d['EnPron'], style)
				sheet.write(row+1, col, '\n'.join(d['means']), style)
				col += 1
			row += 2
		print('Finished.')
		book.save('.'.join(file.split('.')[: -1]) + '.xls')
pass

