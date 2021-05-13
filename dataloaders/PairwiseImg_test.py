from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize
import scipy.misc 
import random

#from dataloaders.helpers import *
from torch.utils.data import Dataset

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale_im(img_temp,scale):
    new_dims = (int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale))
    return cv2.resize(img_temp,new_dims).astype(float)

def scale_gt(img_temp,scale):
    new_dims = (int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale))
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)

def my_crop(img,gt):
    H = int(0.9 * img.shape[0])
    W = int(0.9 * img.shape[1])
    H_offset = random.choice(range(img.shape[0] - H))
    W_offset = random.choice(range(img.shape[1] - W))
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)
    img = img[H_slice, W_slice, :]
    gt = gt[H_slice, W_slice]
    
    return img, gt

class PairwiseImg(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 #+++        db_root_dir='/data/aacunzo/DAVIS-2016',
                 db_root_dir='/thecube/students/lpisaneschi/ILSVRC2017_VID/ILSVRC',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None,
                 sample_range=10):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.range = sample_range
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs2'

        if self.seq_name is None: #Tutti i set di dati partecipano alla formazione
            #+++     with open(os.path.join(db_root_dir, fname + '.txt')) as f:
            #+++     with open("home/aacunzo/COSNet/val_seqs2.txt") as f:
            with open(os.path.join("/home/aacunzo/COSNet", fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                Index = {}
                for seq in seqs:

                    #+++     images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip('\n'))))
                    #+++     images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))

                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'Data/VID/val/', seq.strip('\n'))))
                    images_path = list(map(lambda x: os.path.join('Data/VID/val/', seq.strip(), x), images))
                    start_num = len(img_list)
                    img_list.extend(images_path)
                    end_num = len(img_list)
                    Index[seq.strip('\n')]= np.array([start_num, end_num])

                    #+++     lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip('\n'))))
                    #+++     lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))

                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/VID/val/', seq.strip('\n'))))
                    lab_path = list(map(lambda x: os.path.join('Annotations/VID/val/', seq.strip(), x), lab))
                    labels.extend(lab_path)
        else: #Per tutti gli esempi di addestramento, img_list memorizza il percorso dell'immagine

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, str(seq_name))))
            img_list = list(map(lambda x: os.path.join(( str(seq_name)), x), names_img))
            #name_label = np.sort(os.listdir(os.path.join(db_root_dir,  str(seq_name))))
            labels = [os.path.join( (str(seq_name)+'/saliencymaps'), names_img[0])]
            labels.extend([None]*(len(names_img)-1)) #Aggiungi l'elemento Nessuno dopo l'elenco delle etichette
            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        self.Index = Index
        #img_files = open('all_im.txt','w+')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        target, target_gt,sequence_name = self.make_img_gt_pair(idx) #Telai da dividere durante il test
        target_id = idx
        seq_name1 = self.img_list[target_id].split('/')[-2] #Ottieni il nome del video
        sample = {'target': target, 'target_gt': target_gt, 'seq_name': sequence_name, 'search_0': None}
        if self.range>=1:
            my_index = self.Index[seq_name1]
            search_num = list(range(my_index[0], my_index[1]))  
            search_ids = random.sample(search_num, self.range)#min(len(self.img_list)-1, target_id+np.random.randint(1,self.range+1))
        
            for i in range(0,self.range):
                search_id = search_ids[i]
                search, search_gt,sequence_name = self.make_img_gt_pair(search_id)
                if sample['search_0'] is None:
                    sample['search_0'] = search
                else:
                    sample['search'+'_'+str(i)] = search
            #np.save('search1.npy',search)
            #np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname
       
        else:
            img, gt = self.make_img_gt_pair(idx)
            sample = {'image': img, 'gt': gt}
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

        return sample  #Il risultato finale di questa classe

    def make_img_gt_pair(self, idx): #Il significato di questa funzione è servire la funzione getitem
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]), cv2.IMREAD_COLOR)
        print("self.labels[idx] : " , self.labels[idx])
        print("self.train : ", self.train)
        if self.labels[idx] is not None and self.train:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), cv2.IMREAD_GRAYSCALE)
            #print(os.path.join(self.db_root_dir, self.labels[idx]))
            print("qui")
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            
         ## L'immagine e la corrispondente verità fondamentale sono state lette e può essere eseguito l'aumento dei dati.
        if self.train:  #scaling, cropping and flipping
             img, label = my_crop(img,label)
             scale = random.uniform(0.7, 1.3)
             flip_p = random.uniform(0, 1)
             img_temp = scale_im(img,scale)
             img_temp = flip(img_temp,flip_p)
             gt_temp = scale_gt(label,scale)
             gt_temp = flip(gt_temp,flip_p)
             
             img = img_temp
             label = gt_temp
             
        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            print('ok1')
            #+++     scipy.misc.imsave('label.png',label)
            scipy.misc.imsave('img.png',img)
            if self.labels[idx] is not None and self.train:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        #img = img[:, :, ::-1]
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))        
        img = img.transpose((2, 0, 1))  # NHWC -> NCHW
        
        if self.labels[idx] is not None and self.train:
                gt = np.array(label, dtype=np.int32)
                gt[gt!=0]=1
                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        sequence_name = self.img_list[idx].split('/')[2]
        return img, gt, sequence_name 

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))
        
        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    #dataset = DAVIS2016(db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2016',
                       # train=True, transform=transforms)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
#
#    for i, data in enumerate(dataloader):
#        plt.figure()
#        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
#        if i == 10:
#            break
#
#    plt.show(block=True)
