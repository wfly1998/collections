# -*- coding: utf-8 -*-

import time
import clipboard

from translater import Translater

if __name__ == '__main__':
    last = ''
    while True:
        s = clipboard.paste()
        if s != last:
            s = s.replace('-\n', '')
            s = s.replace('\n', ' ')
            res = Translater.translateSentense(s)
            if len(s) < 50:
                print('%s -> %s' % (s, res))
            else:
                print('Translatation finished.')
            clipboard.copy(res)
            last = res
            continue
        last = s
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            exit(0)
