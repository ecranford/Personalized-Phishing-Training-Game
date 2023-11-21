import os
import sys

files = os.listdir('./img/')
cwd = os.path.realpath('./img/')

for i in files:
    if "2021" in i:
        temp = i.split('_Wed_')[0] + i.split('2021')[1]
        oldpath = cwd + "/" + i
        newpath = cwd + "/" + temp
        os.rename(oldpath, newpath)