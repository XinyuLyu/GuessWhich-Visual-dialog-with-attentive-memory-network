import json
import os
import _pickle as cPickle
import csv
import sys
import csv
import h5py
import numpy as np


'''
# val_imgid2idx.pkl, train_imgid2idx.pkl, test2015_imgid2idx.pkl
# train.hdf5   test2015.hdf5 val.hdf5
# test 14
# train 15
# val 13
fileList = []
for filename in os.listdir(r'/hhd/lvxinyu/data/data/coco_image/val2014/'):
    if filename.endswith('jpg'):
        fileList.append(filename[13:])

fileListidx = []
for i in fileList:
    for j in range(len(i)):
        if i[j] !='0':
            fileListidx.append(i[j:-4])
            break

name = 'val'
dataroot = r'/hhd/liyangzhang/ban-vqa/data/'
img_id2idx = cPickle.load(
    open(os.path.join(dataroot, '%s_imgid2idx.pkl' % (name)), 'rb'))

fileListidx_=[]
for keys in img_id2idx.keys():
    fileListidx_.append(str(keys))

i=0
for ids in fileListidx:
    if ids not in fileListidx_:
        i+=1

print(i)
'''
'''
infile = r'/hhd/lvxinyu/data/coco2015/test2015_36/test2015_resnet101_faster_rcnn_genome_36.tsv'
list =[]
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
with open(infile, "r+") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
    for item in reader:
        image_id = int(item['image_id'])
        features = item['features']
        #item['features'] = bytes(item['features'], 'utf')
list.append(image_id)
print(len(list))
'''
'''
h5_path =r'/hhd/liyangzhang/backup/original_h5file/test2015.hdf5'# (2566887,2048)
h5_path1 =r'/hhd/liyangzhang/backup/BAN_full/data/test201536.hdf5'# (81434,36)
with h5py.File(h5_path1, 'r') as hf:
    features = np.array(hf.get('image_features'))
print()'''

#/hhd/fankaixuan/features/mscoco/bottom_up
#/hhd/fankaixuan/features/mscoco/test_2014

'''
for filename in os.listdir(r'/hhd/fankaixuan/features/mscoco/bottom_up/'):
    if filename.endswith('npz'):
        npy = np.load(r'/hhd/fankaixuan/features/mscoco/bottom_up/'+filename)['feat']
        #np.save(,npy)
        print()
'''
'''


import numpy as np
import h5py

data_to_write = np.random.random(size=(100,20)) # or some such

with h5py.File('name-of-file.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=data_to_write)
'''
'''
i=0
for filename in os.listdir(r'/hhd/fankaixuan/features/mscoco/bottom_up/'):
    image_id = int(filename.split('.')[0])
    np.save(r'/hhd/lvxinyu/data/data/coco_image_new/'+str(image_id)+'.npy',np.load(r'/hhd/fankaixuan/features/mscoco/bottom_up/' + filename)['feat'])
    i+=1
    print((i/123287) *100)
'''
'''
imgFile = h5py.File('/hhd/lvxinyu/aqm_plus/resources/data/visdial_0.9/data_img.h5', 'r')
Feats = np.array(imgFile['images_' +'val'])
'''
'''
train = []
val = []
test = []
with open('/hhd/lvxinyu/aqm_plus/resources/data/visdial_0.9/chat_processed_params.json',
          'r') as fileId:  # spilt image names, word2ind, ind2word
    info = json.load(fileId)
    for i in info['unique_img_train']:
        train.append(int(i.split('_')[2].split('.')[0])) #80000
    for i in info['unique_img_val']:
        val.append(int(i.split('_')[2].split('.')[0]))   #2783
    for i in info['unique_img_test']:
        test.append(int(i.split('_')[2].split('.')[0]))  #40504

train_feature = [None]* 80000
val_feature = [None]* 2783
test_feature = [None]* 41404

i=0
#for filename in os.listdir(r'/hhd/fankaixuan/features/mscoco/bottom_up/'):
for filename in os.listdir(r'/hhd/lvxinyu/data/data/coco_image_new/'):
    #if filename.endswith('npz'):
        image_id = int(filename.split('.')[0])
        npy = np.load(r'/hhd/lvxinyu/data/data/coco_image_new/'+filename).tolist()
        #npz = np.load(r'/hhd/fankaixuan/features/mscoco/bottom_up/' + filename)['feat'].tolist()
        if i < 40000:
            pass
        elif image_id in train:
            train_feature[train.index(image_id)]= npy
            #index = train.index(image_id)
            #train_feature[index] = npz
        elif image_id in val:
            val_feature[val.index(image_id)] = npy
            #index = val.index(image_id)
            #val_feature[index] = npz
        elif image_id in test:
            test_feature[test.index(image_id)] = npy
            #index = test.index(image_id)
            #test_feature[index] = npz
        #else:
        #    pass
        i+=1

        print((i/80000) *100) #123287
train_feature_np = np.array(train_feature)
val_feature_np = np.array(val_feature)
test_feature_np = np.array(test_feature)
np.save('/hhd/lvxinyu/data/data/coco_image_new/train1.npy',train_feature_np)
np.save('/hhd/lvxinyu/data/data/coco_image_new/val1.npy',val_feature_np)
np.save('/hhd/lvxinyu/data/data/coco_image_new/test1.npy',test_feature_np)

print()
'''

train = []
val = []
test = []
train_dict = {}
test_dict = {}
val_dict = {}
with open('/hhd/lvxinyu/aqm_plus/resources/data/visdial_0.9/chat_processed_params.json',
          'r') as fileId:  # spilt image names, word2ind, ind2word
    info = json.load(fileId)
    for i in info['unique_img_train']:
        train.append(int(i.split('_')[2].split('.')[0])) #80000
        #train_dict[len(train) - 1] = int(i.split('_')[2].split('.')[0])
    for i in info['unique_img_val']:
        val.append(int(i.split('_')[2].split('.')[0]))   #2783
        #val_dict[len(val) - 1] = int(i.split('_')[2].split('.')[0])
    for i in info['unique_img_test']:
        test.append(int(i.split('_')[2].split('.')[0]))  #41404
        #test_dict[len(test) - 1] = int(i.split('_')[2].split('.')[0])

np.save(r'/hhd/lvxinyu/data/data/coco_image_new_split/train_image_map.npy',train)
np.save(r'/hhd/lvxinyu/data/data/coco_image_new_split/val_image_map.npy',val)
np.save(r'/hhd/lvxinyu/data/data/coco_image_new_split/test_image_map.npy',test)

'''

train_feature = np.load('/hhd/lvxinyu/data/data/coco_image_new/train1.npy')
val_feature = np.load('/hhd/lvxinyu/data/data/coco_image_new/val1.npy')
test_feature = np.load('/hhd/lvxinyu/data/data/coco_image_new/test1.npy')
'''
'''
train_feature = [None]* 80000
val_feature = [None]* 2783
test_feature = [None]* 40504
i=0
#for filename in os.listdir(r'/hhd/fankaixuan/features/mscoco/bottom_up/'):
for filename in os.listdir(r'/hhd/lvxinyu/data/data/coco_image_new/'):
    #if filename.endswith('npz'):
        image_id = int(filename.split('.')[0])
        npy = np.load(r'/hhd/lvxinyu/data/data/coco_image_new/'+filename).tolist()
        #npz = np.load(r'/hhd/fankaixuan/features/mscoco/bottom_up/' + filename)['feat'].tolist()
        if i < 40000:
            pass
        elif i == 80000:
            break
        #elif image_id in train:
        #    train_feature[train.index(image_id)]= npy
            #index = train.index(image_id)
            #train_feature[index] = npz
        #elif image_id in val:
        #    val_feature[val.index(image_id)] = npy
            #index = val.index(image_id)
            #val_feature[index] = npz
        elif image_id in test:
            test_feature[test.index(image_id)] = npy
            #index = test.index(image_id)
            #test_feature[index] = npz
        #else:
        #    pass
        i+=1

        print((i/80000) *100) #123287
#train_feature_np = np.array(train_feature)
#val_feature_np = np.array(val_feature)
test_feature_np = np.array(test_feature)
#np.save('/hhd/lvxinyu/data/data/coco_image_new/train2.npy',train_feature_np)
#np.save('/hhd/lvxinyu/data/data/coco_image_new/val2.npy',val_feature_np)
np.save('/hhd/lvxinyu/data/data/coco_image_new/test2.npy',test_feature_np)

print()
'''

'''
train_feature1 = np.load('/hhd/lvxinyu/data/data/coco_image_new/train_complete1.npy')
print('1')
#train_feature2 = np.load('/hhd/lvxinyu/data/data/coco_image_new_split/train2.npy')
#print('2')
train_feature3 = np.load('/hhd/lvxinyu/data/data/coco_image_new_split/train3.npy')
print('3')
for i in range(len(train_feature1)):
    if train_feature1[i] is None:
        #if train_feature2[i] is not None:
        #    train_feature1[i] = train_feature2[i]
        if train_feature3[i] is not None:
            train_feature1[i] = train_feature3[i]
    else:
        pass
print('save')
np.save('/hhd/lvxinyu/data/data/coco_image_new/train_complete.npy',train_feature1)
print('finish save')
'''
'''
arr1 = np.random.randn(80000,36,2048)
train_feature1 = np.load('/hhd/lvxinyu/data/data/coco_image_new/train_complete.npy')
for i in range(len(train_feature1)):
    arr1[i]=np.array(train_feature1[i])
print('convert to numpy finish')
np.save(r'/hhd/lvxinyu/data/data/coco_image_new_split/train.npy',arr1)
print('finish save')
'''

'''
import h5py
import numpy as np
f=h5py.File(r'/home/lvxinyu/myh5py.hdf5',"r")
a=np.arange(20)
d1=f.create_dataset('images_val', data=a)
for key in f.keys():
    print(f[key].name)
    print(f[key].name)
    print(f[key].value)
    '''

