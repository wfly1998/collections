# -*- coding: utf-8 -*-

import clipboard

from translater import Translater

if __name__ == '__main__':
    while True:
        try:
            s = input(' > ')
        except KeyboardInterrupt:
            exit(0)
        if len(s) == 0:
            s = clipboard.paste()
        s = s.replace('-\n', '')
        s = s.replace('\n', ' ')
        res = Translater.translateSentense(s)
        clipboard.copy(res)
        print(res)
