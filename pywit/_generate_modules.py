'''
generate files to mirror in pywit the modules from xwakes.wit
'''

import os

fff = os.listdir('../xwakes/wit')

for nn in fff:
    if not nn.endswith('.py') or nn.startswith('_'):
        continue

    with open(nn, 'w') as fid:
        fid.write(f'from xwakes.wit.{nn.split(".py")[0]} import *\n')