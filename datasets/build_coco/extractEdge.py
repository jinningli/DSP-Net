import os
import cv2
import numpy as np
import argparse
from os.path import join
from random import shuffle
import os.path as path


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, type=str)
args = parser.parse_args()

def getedge(in_dir, out_dir, para1=180, para2=300):
    img = cv2.imread(in_dir,0)
    edges = cv2.Canny(img, para1, para2)
    cv2.imwrite(out_dir, edges)

def copy(f, t):
    os.system("cp " + f + " " + t)
    # print("cp " + f + " " + t)

def mkdir(path):
    if not os.path.exists(path):
        os.system('mkdir ' + path)
        print('mkdir '  + path)

"""
Put real image in real, input dataset root
"""

def distribute_A(test_weight = 0.1):
    cnt = 0
    mkdir(join(args.dataroot, 'trainA'))
    mkdir(join(args.dataroot, 'testA'))
    for root, dirs, files in sorted(os.walk(os.path.join(args.dataroot, 'real'))):
        test_cnt = int(len(files) * test_weight)
        shuffle(files)
        tot = len(files)
        for file in files:
            if file.find(".jpg")==-1:
                continue
            cnt += 1
            if cnt % 2000 == 0:
                print(str(cnt) + '/' + str(tot))
            if cnt < test_cnt:
                input_dir = os.path.join(root, file)
                out_dir = input_dir.replace('real', 'test' + 'A')
                print(input_dir + ' ---> ' + out_dir)
                getedge(input_dir, out_dir)
            else:
                input_dir = os.path.join(root, file)
                out_dir = input_dir.replace('real', 'train' + 'A')
                # print(input_dir + ' ---> ' + out_dir)
                getedge(input_dir, out_dir)

distribute_A()

"""
Put semantic image in semantic
"""

def distribute_BC():
    for phase in ['train', 'test']:
        mkdir(path.join(args.dataroot, phase + 'B'))
        mkdir(path.join(args.dataroot, phase + 'C'))
        cnt = 0
        for root, dirs, files in os.walk(path.join(args.dataroot, phase + 'A')):
            tot = len(files)
            for file in files:
                if file.find(".jpg") == -1:
                    continue
                cnt += 1
                if cnt % 2000 == 0:
                    print(str(cnt) + '/' + str(tot))
                Bfrom_path = path.join(root, file).replace(phase + 'A', 'painting')
                Bto_path = Bfrom_path.replace('painting', phase + 'B')
                copy(Bfrom_path, Bto_path)
                Cfrom_path = path.join(root, file).replace(phase + 'A', 'semantic')
                Cto_path = Cfrom_path.replace('semantic', phase + 'C')
                copy(Cfrom_path, Cto_path)

distribute_BC()
