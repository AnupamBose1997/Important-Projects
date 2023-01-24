import json
from updated import MergeFile


with open('C:/Users/ASUS/Documents/visual studio/path.json') as f:
    pathconfig = json.load(f)

x = MergeFile(pathconfig['path']['masterpath'],
              pathconfig['path']['outputpath'])
x.writeintofile()
