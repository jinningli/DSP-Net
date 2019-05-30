import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pylab
import os
from os.path import join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, type=str)
parser.add_argument('--annpath', required=True, type=str)
args = parser.parse_args()

dataroot = args.dataroot
annFile=args.annpath

coco=COCO(annFile)

for root, dirs, files in os.walk(join(dataroot, 'real')):
    for file in files:
        if file.find('.jpg') == -1:
            continue
        n = file.replace('.jpg', '')
        number = int(n)
        img = coco.loadImgs(ids=number)[0]
        height = img['height']
        width = img['width']

        plt.rcParams['figure.figsize'] = (width, height)
        plt.rcParams['savefig.dpi'] = 1
        plt.rcParams['figure.dpi'] = 1

        im = Image.new('RGB', (width, height), 'white')

        plt.axis('off')
        plt.imshow(im)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        # plt.show()
        if not os.path.exists(join(dataroot, 'semantic')):
            os.mkdir(join(dataroot, 'semantic'))
        plt.savefig(join(dataroot, 'semantic', file.replace('.jpg', '.png')), box_inches='tight', format='png')
        Image.open(join(dataroot, 'semantic', file.replace('.jpg', '.png'))).convert('RGB').save(join(dataroot, 'semantic', file), 'JPEG')
        # print('cp ' + join(root, file) + join(' images', 'tagged', file.replace('.jpg', '_.jpg')))
        # os.system('cp ' + join(root, file) + join(' images', 'tagged', file.replace('.jpg', '_.jpg')))
        plt.close()
        print(join(dataroot, 'semantic', file))
