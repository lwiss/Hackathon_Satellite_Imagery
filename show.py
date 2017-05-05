import os
import cv2
import sys
import scipy.misc
import cPickle
import numpy as np

if len(sys.argv)!=2:
    print('Usage: python show.py <folder>')
    exit(0)

folder=sys.argv[1]
gt_suf=[]
inp_suf=[]
pred_suf=[]
for file_name in os.listdir(folder):
    file_name='.'.join(file_name.split('.')[:-1])
    if len(file_name)>2 and file_name[:2]=='gt':
        gt_suf.append(file_name[2:])
    elif len(file_name)>3 and file_name[:3]=='inp':
        inp_suf.append(file_name[3:])
    elif len(file_name)>4 and file_name[:4]=='pred':
        pred_suf.append(file_name[4:])

common_suf=[]
for suf in gt_suf:
    if suf in inp_suf and suf in pred_suf:
        common_suf.append(suf)
print('%s files detected'%len(common_suf))

def bw2color(data):
    if len(data.shape)==2:
        data=data.reshape([data.shape[0],data.shape[1],1])
        data=np.concatenate([data,data,data],axis=2)
    return data

for suf in common_suf:
    ori_image=folder+os.sep+'inp%s.tif'%suf
    pred_image=folder+os.sep+'pred%s.tif'%suf
    gt_image=folder+os.sep+'gt%s.tif'%suf
    ori_data=scipy.misc.imread(ori_image)
    pred_data=scipy.misc.imread(pred_image)
    gt_data=scipy.misc.imread(gt_image)
    cmp_data=[]
    for pred,gt in zip(pred_data.reshape(-1),gt_data.reshape(-1)):
        if pred>128 and gt>128:
            cmp_data.append([0,255,0])
        elif pred>128 and gt<128:
            cmp_data.append([255,0,0])
        elif pred<128 and gt>128:
            cmp_data.append([0,255,255])
        else:
            cmp_data.append([0,0,0])
    cmp_data=np.array(cmp_data).reshape([pred_data.shape[0],pred_data.shape[1],3])
    pred_data=bw2color(pred_data)
    gt_data=bw2color(gt_data)
    up_half=np.concatenate([ori_data,gt_data],axis=1)
    down_half=np.concatenate([pred_data,cmp_data],axis=1)
    final=np.concatenate([up_half,down_half],axis=0)
    final=final.astype(np.uint8)
    cPickle.dump(final,open('final.pkl','wb'))
    while True:
        cv2.imshow('image',final)
        key=cv2.waitKey(1) & 0xFF

        if key==ord('c'):
            break
        elif key==ord('z'):
            exit(0)
