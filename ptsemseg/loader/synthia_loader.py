import matplotlib
matplotlib.use('Agg')

import os
import torch
import numpy as np
import scipy.misc as m
import glob
import cv2
import time
import matplotlib.pyplot as plt
import copy
from random import shuffle
import random
from torch.utils import data
import yaml
from tqdm import tqdm
import pickle



class synthiaLoader(data.Dataset):

    class_names = np.array([
        "void",
        "sky",
        "building",
        "road",
        "sidewalk",
        "fence",
        "vegetation",
        "pole",
        "car",
        "traffic sign",
        "pedestrian",
        "bicycle",
        "lanemarking",
        "X",
        "Y",
        "traffic light"
    ])
    
    image_modes = ['RGB', 'Depth', 'GT/COLOR', 'GT/LABELS']
    sides = ['Stereo_Left','Stereo_Right']
    cam_pos = ['Omni_B','Omni_F','Omni_L','Omni_R']

    split_subdirs = {}
    ignore_index = 0
    mean_rgbd = {
        "synthia-seq": [55.09944,  62.203827, 71.23802 , 130.35643,1.8075644,15.805721] # synthia-seq
    } 
    std_rgbd = {
        "synthia-seq": [49.56111,  51.497387, 55.363934 , 46.930763, 10.479317, 34.19771] # synthia-seq   
    }
    
   
    def __init__(
        self,
        root,
        split="train",
        subsplits=None,
        is_transform=True,
        img_rows=512,
        img_cols = 512,
        reduction=1.0,      
        img_norm=True,
        augmentations = None,
        version='synthia-seq'   
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """

        self.root = root
        self.split = split
        self.subsplits = subsplits
        self.is_transform = is_transform
        self.img_norm = img_norm
        self.n_classes = len(self.class_names)
        self.img_size = (img_rows,img_cols)
        
        # split: train/val image_modes
        self.imgs = {image_mode:[] for image_mode in self.image_modes}
        self.dgrd = {image_mode:[] for image_mode in self.image_modes}
        self.mean = np.array(self.mean_rgbd[version])
        self.std = np.array(self.std_rgbd[version])

        self.n_classes = 16
        # load RGB/Depth
        for subsplit in self.subsplits:
            if len(subsplit.split("__")) == 2:
                condition = subsplit.split("__")[0]
            else:
                condition = subsplit
            for comb_modal in self.image_modes:
                for comb_cam in self.cam_pos:
                    for side in self.sides:
                        files = glob.glob(os.path.join(root,condition,comb_modal,side,comb_cam,'*.png'),recursive=True)
                        random.seed(0)
                        shuffle(files)
                        n = len(files)
                        n_train = int(0.7 * n)
                        n_valid = int(0.1 * n)
                        n_test = int(0.2 * n)
                        
                        if self.split == 'train':
                            files = files[:n_train]
                        if self.split == 'val':
                            files = files[n_train:n_train+n_valid]
                        if self.split == 'test':
                            files = files[n_train+n_valid:]
                        for file_path in files:
                            self.imgs[comb_modal].append(file_path)
        
        if not self.imgs[self.image_modes[0]]:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))
        print("{} {}: Found {} Images".format(self.split,self.subsplits,len(self.imgs[self.image_modes[0]])))
        if reduction != 1.0:
            for image_mode in self.image_modes:
                self.imgs[image_mode] = self.imgs[image_mode][::int(1/reduction)]
            print("{} {}: Reduced by {} to {} Images".format(self.split,self.subsplits,reduction,len(self.imgs[self.image_modes[0]])))


    def tuple_to_folder_name(self, path_tuple):
        start = path_tuple[1]
        end = path_tuple[2]
        path=str(start[0])+'_'+str(-start[1])+'__'+str(end[0])+'_'+str(-end[1])
        return path


    def __len__(self):
        """__len__"""
        return len(self.imgs[self.image_modes[0]])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        start_ts = time.time()

        img_path = self.imgs['RGB'][index]
        lbl_path = self.imgs['GT/LABELS'][index]
        
        img_raw = np.array(cv2.imread(img_path),dtype=np.uint8)[:,:,:3]
        lbl = np.array(cv2.imread(lbl_path,cv2.IMREAD_UNCHANGED))[:,:,2]
        
        if self.is_transform:
            img, lbl = self.transform(img_raw, lbl)
        return img, lbl



    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        # if img.dtype == 'uint8':   
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img.astype(np.float64)
    
        if self.img_norm:
            img = np.divide((img.astype(float) - self.mean[:3]),self.std[:3])
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST) #, "nearest", mode="F")
        lbl = lbl.astype(int)

        # if not np.all(classes == np.unique(lbl)):
        #     print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    # def get_cls_num_list(self):
    #     cls_num_list = []
    #     cls_num_dict = Counter(self.classes)
    #     for k in range(max(self.classes)):
    #         cls_num_list.append(cls_num_dict[k])
    #     return cls_num_list

if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt


    local_path = "/home/datasets/synthia-seq/"

    dst = airsimLoader(local_path, is_transform=True,split='val') #, augmentations=augmentations)
    bs = 4

    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels  = data
        
        # import pdb;pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(labels.numpy()[j])
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()