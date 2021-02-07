#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from ptsemseg.loader.transform import *



class cityscapesLoader(Dataset):
    def __init__(self, root, img_rows = 769, img_cols = 769, split='train', label_scale=1.0, *args, **kwargs):
        super(cityscapesLoader, self).__init__(*args, **kwargs)
        assert split in ('train', 'val', 'test')
        img_size = (img_rows,img_cols)
        self.mode = split
        self.n_classes = 19
        self.ignore_lb = 255
        self.label_scale = label_scale
        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(root, 'leftImg8bit', split)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        if self.mode =='test':
            gtpth = osp.join(root, split)
        else:
            gtpth = osp.join(root, 'gtFine', split)
        folders = os.listdir(gtpth)
        for fd in folders:
            if fd == 'info.json':
                continue

            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        if self.mode != 'test':
            assert set(imgnames) == set(gtnames)
            assert set(self.imnames) == set(self.imgs.keys())
            assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(img_size)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        img = Image.open(impth)

        if self.mode != 'test':
            lbpth = self.labels[fn]
            label = Image.open(lbpth)

        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']

            img = self.to_tensor(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]
            label = self.convert_labels(label)
            return img, label
        elif self.mode == 'val':
            img = self.to_tensor(img)

            if self.label_scale != 1.0:
                H,W = label.size
                label = label.resize((int(H*self.label_scale), int(W*self.label_scale)),
                                      Image.NEAREST)

            label = np.array(label).astype(np.int64)[np.newaxis, :]
            label = self.convert_labels(label)
            return img, label
        else:
            img = self.to_tensor(img)
            return img, impth



    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))

