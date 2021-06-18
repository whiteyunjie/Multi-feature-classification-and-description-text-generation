import os
import numpy as np
import json



ORIPATH = '/data/images/pathology/temp/Qingdao'

dirs = os.listdir(ORIPATH)
data = []
for path in dirs:
    if os.path.isdir(os.path.join(ORIPATH,path)):
        xmlpath = os.path.join(ORIPATH,path,path+'_tumor.xml')
        print(xmlpath)
        if os.path.exists(xmlpath):
            wsipath = os.path.join(ORIPATH,path)
            data.append(wsipath)
print('in total:',str(len(data)))
print('saving')
np.save('datalist.npy',data)

'''
dirs = os.listdir(ORIPATH)
pidlist = np.load('pidlist.npy',allow_pickle=True)
plist = {}
slist = []
lacklist = []
for path in pidlist:
    exist = False
    if os.path.isdir(os.path.join(ORIPATH,path)):
        #xmlpath = os.path.join(ORIPATH,path,path+'_tumor.xml')
        #print(xmlpath)
        slist.append(path)
        if path not in plist:
            plist[path] = []
            plist[path].append(path)
        else:
            plist[path].append(path)
        exist = True
    else:
        for i in range(10):
            spath = path + '-' + str(i)
            if os.path.isdir(os.path.join(ORIPATH,spath)):
                slist.append(spath)
                if path not in plist:
                    plist[path] = []
                    plist[path].append(spath)
                else:
                    plist[path].append(spath)
                exist = True
    if not exist:
        print(f'lack of {path}')
        lacklist.append(path)

with open('plist.json', 'w') as f:
    json.dump(plist, f)
print('in total:',str(len(slist)))
print('in total patients:',str(len(plist)))
np.save('slist.npy',slist)
np.save('lacklist.npy',lacklist)
'''
