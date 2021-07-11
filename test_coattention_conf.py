import datetime
import argparse
import torch
import torch.nn as nn
from motmetrics import distances
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from dataloaders import PairwiseImg_test as db
#from dataloaders import StaticImg as db #采用voc dataset的数据设置格式方法
import matplotlib.pyplot as plt
import random
import timeit
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
#from utils.colorize_mask import cityscapes_colorize_mask, VOCColorize
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from deeplab.siamese_model_conf import CoattentionNet
from torchvision.utils import save_image
from torchvision import transforms

import box_tracker

import motmetrics as mm
import xml.etree.ElementTree as ET



def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description="PSPnet")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        help="voc12, cityscapes, or pascal-context")

    # GPU configuration
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="0",
                        help="choose gpu device.")
    parser.add_argument("--seq_name", default = 'bmx-bumps')
    parser.add_argument("--use_crf", default = 'True')
    parser.add_argument("--sample_range", default =5)
    parser.add_argument("--tresh", default=127)
    parser.add_argument("--step", default=1)
    parser.add_argument("--mode", default='normal')
    parser.add_argument("--type", default='1')
    
    return parser.parse_args()

def configure_dataset_model(args):
    if args.dataset == 'voc12':
        args.data_dir ='/home/wty/AllDataSet/VOC2012'  #Path to the directory containing the PASCAL VOC dataset
        args.data_list = './dataset/list/VOC2012/test.txt'  #Path to the file listing the images in the dataset
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) 
        #RBG mean, first subtract mean and then change to BGR
        args.ignore_label = 255   #The index of the label to ignore during the training
        args.num_classes = 21  #Number of classes to predict (including background)
        args.restore_from = './snapshots/voc12/psp_voc12_14.pth'  #Where restore model parameters from
        args.save_segimage = True
        args.seg_save_dir = "./result/test/VOC2012"
        args.corp_size =(505, 505)

    elif args.dataset == 'imagenet':
        args.batch_size = 1 # 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        #args.data_dir = '/thecube/students/lpisaneschi/ILSVRC2017_VID/ILSVRC'  # 37572 image pairs
        #args.data_list = '/thecube/students/lpisaneschi/ILSVRC2017_VID/ILSVRC/val_seqs1.txt'  # Path to the file listing the images in the dataset

        args.data_dir = '/mnt/ILSVRC2017_VID/ILSVRC'
        args.data_list = '/mnt/ILSVRC2017_VID/ILSVRC/ImageSets/VID/val.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '473,473' #Comma-separated string with height and width of images
        #args.input_size = '1280,720'  # Comma-separated string with height and width of images
        args.num_classes = 2      #Number of classes to predict (including background) ****
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        args.restore_from = './co_attention.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = './snapshots/imagenet_iteration/'          #Where to save snapshots of the model
        args.save_segimage = True
        args.seg_save_dir = "./result/test/imagenet_iteration_conf"
        args.vis_save_dir = "./result/test/imagenet_vis"
        args.corp_size =(473, 473)
        #args.corp_size = (640, 360)

    elif args.dataset == 'davis':
        args.batch_size = 1 # 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.data_dir = '/data/aacunzo/DAVIS-2016'   # 37572 image pairs
        args.data_list = '/DAVIS-2016/val_seqs1.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '473,473' #Comma-separated string with height and width of images
        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        args.restore_from = './co_attention.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = './snapshots/davis_iteration/'          #Where to save snapshots of the model
        args.save_segimage = True
        args.seg_save_dir = "./result/test/davis_iteration_conf"
        args.vis_save_dir = "./result/test/davis_vis"
        args.corp_size =(473, 473)

    elif args.dataset == 'davis_yoda':
        args.batch_size = 1 # 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.data_dir = '/home/aacunzo/DAVIS-2016'   # 37572 image pairs
        args.data_list = './val_seqs1.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '473,473' #Comma-separated string with height and width of images
        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        args.restore_from = './co_attention.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = './snapshots/davis_iteration/'          #Where to save snapshots of the model
        args.save_segimage = True
        args.seg_save_dir = "./result/test/davis_iteration_conf"
        args.vis_save_dir = "./result/test/davis_vis"
        args.corp_size =(473, 473)
        
    else:
        print("dataset error")

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
       You probably saved the model using nn.DataParallel, which stores the model in module, and now you are trying to load it 
       without DataParallel. You can either add a nn.DataParallel temporarily in your network for loading purposes, or you can 
       load the weights file, create a new ordered dict without the module prefix, and load it back 
    """
    state_dict_new = OrderedDict()
    #print(type(state_dict))
    for k, v in state_dict.items():
        #print(k)
        name = k[7:] # remove the prefix module.
        # My heart is broken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
        if name == 'linear_e.weight':
            np.save('weight_matrix.npy',v.cpu().numpy())
    return state_dict_new

def sigmoid(inX): 
    return 1.0/(1+np.exp(-inX))  #Definisci un metodo sigmoideo, la sua essenza è 1 / (1 + e ^ -x)

def main():

    date_for_txt = datetime.datetime.now()
    string_data = str(date_for_txt.day) + "-" + str(date_for_txt.month) + "-" + str(date_for_txt.year) + "-" + str(date_for_txt.hour) + "-" + str(date_for_txt.minute) + "-" + str(date_for_txt.second)

    #string_data = "2-7-2021-10-16-10"

    args = get_arguments()
    print("=====> Configure dataset and model")
    configure_dataset_model(args)
    #print(args)
    model = CoattentionNet(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from, map_location=lambda storage, loc: storage)
    #print(saved_state_dict.keys())
    #model.load_state_dict({k.replace('pspmodule.',''):v for k,v in torch.load(args.restore_from)['state_dict'].items()})
    model.load_state_dict(convert_state_dict(saved_state_dict["model"])) #convert_state_dict(saved_state_dict["model"])

    model.eval()
    model.cuda()

    if args.dataset == 'imagenet':  #for imagenet
        db_test = db.PairwiseImg(train=False, inputRes=(473, 473), db_root_dir=args.data_dir, transform=None,seq_name=None, sample_range=args.sample_range)
        #db_test = db.PairwiseImg(train=False, inputRes=(640,360), db_root_dir=args.data_dir,  transform=None, seq_name = None, sample_range = args.sample_range)
        #db_test = db.PairwiseImg(train=False, inputRes=None, db_root_dir=args.data_dir,  transform=None, seq_name = None, sample_range = args.sample_range) #db_root_dir() --> '/path/to/DAVIS-2016' train path
        testloader = data.DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
        #voc_colorize = VOCColorize()

    elif args.dataset == 'davis' or  args.dataset == 'davis_yoda': #for davis 2016
        db_test = db.PairwiseImg(train=False, inputRes=(473,473), db_root_dir=args.data_dir,  transform=None, seq_name = None, sample_range = args.sample_range) #db_root_dir() --> '/path/to/DAVIS-2016' train path
        testloader = data.DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
        # voc_colorize = VOCColorize()
    else:
        print("dataset error")

    data_list = []

    if args.save_segimage:
        if not os.path.exists(args.seg_save_dir) and not os.path.exists(args.vis_save_dir):
            os.makedirs(args.seg_save_dir)
            os.makedirs(args.vis_save_dir)

    print("======> test set size:", len(testloader))
    old_temp=''

    img_sequencies_name = []
    #soglia = 127
    soglia = int(args.tresh)
    print("soglia: ",soglia)

    if args.dataset == 'davis' or args.dataset == 'davis_yoda':
        f_val_seq = open("./val_seqs1.txt", "r")
        my_index = 0
    else:
        f_val_seq = open("./val_seqs2.txt", "r")
        my_index = -1
    img_sequencies_name = [x.strip() for x in f_val_seq.readlines()]
    print(img_sequencies_name)
    f_val_seq.close()
    cont = 0
    args.seq_name="0"

    if args.step == '1':
        #'''
        for index, batch in enumerate(testloader):
            print("----------------------------------------------------------------------------------------------------------------------")
            print("processed index: ", '%d/%d processed'%(index,len(testloader)))
            target = batch['target']
            temp = batch['seq_name']

            #print(batch)

            #args.seq_name=temp[0]

            if index==0  and args.data_dir == '/mnt/ILSVRC2017_VID/ILSVRC':
                args.seq_name=img_sequencies_name[0]


            # Creo lista di seq name
            #if not img_sequencies_name.__contains__(args.seq_name):
            #    print("img seq name non contiene " + args.seq_name)
            #    img_sequencies_name.append(args.seq_name)
            #    print("img seq name : " )
            #    print(img_sequencies_name)




            #print("Target : ", target) # Tensor
            #print("Target shape: ", target.shape) # torch.Size([1, 3, 473, 473])
            print("Temp : ", temp) # [blackswan]
            #print("Seq_name : ", args.seq_name) # blackswan

            path_save_img = "./IMG_PROVA"
            filename = os.path.join(path_save_img, 'target_normalized.png')

            img_target = target[0] # torch.Size([3, 473, 473])

            PIL_img = transforms.ToPILImage()(img_target)
            #PIL_img.convert("RGB")
            PIL_img.save(filename)

            img_target_numpy = img_target.numpy()
            img_target_numpy = img_target_numpy.transpose((1, 2, 0))  # CHW --> HWC

            img_target_numpy = img_target_numpy + np.array(db_test.meanval)
            #print("img target numpy denorm: ", img_target_numpy)
            img_target_numpy = img_target_numpy.astype(np.uint8)
            #print("img target numpy : ", img_target_numpy)

            PIL_img_from_numpy = Image.fromarray(img_target_numpy)

            filename_target = os.path.join(path_save_img, 'target_denormalized.png')
            PIL_img_from_numpy.save(filename_target)

            if (args.data_dir == '/data/aacunzo/DAVIS-2016' or args.data_dir == '/home/aacunzo/DAVIS-2016'):
                args.seq_name = temp[0]
                # my_index è l'indice nella seq_name
                print("old_temp: " , old_temp)
                if old_temp==args.seq_name:
                    my_index = my_index+1
                else:
                    my_index = 0

            if (args.data_dir == '/mnt/ILSVRC2017_VID/ILSVRC'):
                path_original_img = os.path.join(args.data_dir, 'Data/VID/val')
                path_original_img = path_original_img + "/" + args.seq_name
                end = len([name for name in os.listdir(path_original_img) if os.path.isfile(os.path.join(path_original_img, name))])
                print("num img :", end)
                print("my_index : ",my_index)
                print("old_temp: ", old_temp)
                my_index = my_index + 1
                if my_index == end :
                    my_index = 0
                    cont = cont+1
                    args.seq_name = img_sequencies_name[cont]

            print("my_index : ",my_index)
            print("cont : ",cont)
            print("seq name : ",args.seq_name)

            output_sum = 0

            #print("Sample_range : ", args.sample_range) # 5



            for i in range(0,args.sample_range):
                search = batch['search'+'_'+str(i)]
                search_im = search

                #print("Search = Search_im : ", search) # Matrice
                #print("Dim search_im : ", search_im.size()) #  torch.Size([1, 3, 473, 473])
                output = model(Variable(target, volatile=True).cuda(),Variable(search_im, volatile=True).cuda())
                #print(output[0]) # output ne ha due
                #print("Output : ", output)

                output_sum = output_sum + output[0].data[0,0].cpu().numpy() #Il risultato della divisione di quel ramo
                #np.save('infer'+str(i)+'.npy',output1)
                #output2 = output[1].data[0, 0].cpu().numpy() #interp'

            output1 = output_sum/args.sample_range


            #if(my_index==0):
            if(args.data_dir == '/data/aacunzo/DAVIS-2016' or args.data_dir == '/home/aacunzo/DAVIS-2016'):
                first_image = np.array(Image.open(args.data_dir+'/JPEGImages/480p/blackswan/00000.jpg'))
                path_annotation = os.path.join(args.data_dir, 'Annotations/480p')
                print("path annotation : " + path_annotation)
                path_original_img = os.path.join(args.data_dir, "JPEGImages/480p")
                path_original_img = path_original_img + "/" + args.seq_name
                print("path original img : " + path_original_img)


            #if (args.data_dir == '/thecube/students/lpisaneschi/ILSVRC2017_VID/ILSVRC'):
            if (args.data_dir == '/mnt/ILSVRC2017_VID/ILSVRC'):
                first_image = np.array(Image.open(args.data_dir + '/Data/VID/val/ILSVRC2015_val_00000000/000000.JPEG'))
                path_annotation = os.path.join(args.data_dir, 'Annotations/VID/val')
                print("path annotation : " + path_annotation)
                path_original_img = os.path.join(args.data_dir, 'Data/VID/val')
                path_original_img = path_original_img + "/" + img_sequencies_name[cont]
                print("path original img : " + path_original_img)


            original_shape = first_image.shape # (480, 854, 3)
            output1 = cv2.resize(output1, (original_shape[1],original_shape[0])) # shape:(480, 854)

            mask = (output1*255).astype(np.uint8)
            mask = Image.fromarray(mask)

            '''
            if args.dataset == 'imagenet':

                save_dir_res = os.path.join(args.seg_save_dir, 'Results-Imagenet'.format(soglia), args.seq_name)
                old_temp = args.seq_name
                if not os.path.exists(save_dir_res):
                    os.makedirs(save_dir_res)
                seg_filename = os.path.join(save_dir_res, 'Mask')
                if not os.path.exists(seg_filename):
                    os.makedirs(seg_filename)
                if args.save_segimage:
                    my_index1 = str(my_index).zfill(6)
                    seg_filename = os.path.join(save_dir_res, 'mask_{}.png'.format(my_index1))
                    mask.save(seg_filename)

                    text_dir = os.path.join(save_dir_res, 'Txt')
                    if not os.path.exists(text_dir):
                        os.makedirs(text_dir)

                    box_text_annotation_filename = os.path.join(text_dir, 'boxes_annotations.txt')
                    
                    img_mask = cv2.imread(seg_filename)
                    img_original = cv2.imread(filename_target)
                    img_original = cv2.resize(img_original, (original_shape[1], original_shape[0]))
                    result_mask = img_mask.copy()
                    result_original = img_original.copy()
                    gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(gray, soglia, 255, cv2.THRESH_BINARY)[1]
                    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    for cntr in contours:
                        x, y, w, h = cv2.boundingRect(cntr)
                        cv2.rectangle(result_mask, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.rectangle(result_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # print("x,y,w,h:", x, y, w, h)

                    # save resulting image
                    cv2.imwrite(os.path.join(save_dir_res, 'BoundingBox_mask_{}.png'.format(my_index1)), result_mask)
                    cv2.imwrite(os.path.join(save_dir_res, 'BoundingBox_img_{}.png'.format(my_index1)), result_original) 
            '''

            save_dir_res = os.path.join(args.seg_save_dir, 'Results_{}'.format(soglia) , args.seq_name)
            old_temp=args.seq_name
            if not os.path.exists(save_dir_res):
                os.makedirs(save_dir_res)
            seg_filename = os.path.join(save_dir_res, 'Mask')
            if not os.path.exists(seg_filename):
                os.makedirs(seg_filename)
            if args.save_segimage:
                if args.dataset == 'davis' or args.dataset == 'davis_yoda':
                    my_index1 = str(my_index).zfill(5)
                else:
                    my_index1 = str(my_index).zfill(6)
                seg_filename = os.path.join(seg_filename, 'mask_{}.png'.format(my_index1))
                #color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
                mask.save(seg_filename)

                # ***
                # take BoundingBox on mask annotation for py-motmetrics
                # File txt for save bbox annotations

                text_dir = os.path.join(save_dir_res, 'Txt')
                if not os.path.exists(text_dir):
                    os.makedirs(text_dir)
                #box_text_annotation_filename = os.path.join(text_dir, 'boxes_annotations_' + string_data + '.txt')
                box_text_annotation_filename = os.path.join(text_dir, 'boxes_annotations.txt')

                #Bounding box di tutte le annotazioni


                if os.path.exists(box_text_annotation_filename):
                    f_annotation = open(box_text_annotation_filename, "a")
                else:
                    f_annotation = open(box_text_annotation_filename, "w")

                path_annotation = os.path.join(path_annotation, args.seq_name)
                print("path annotation : " + path_annotation)
                if args.dataset == 'davis' or args.dataset == 'davis_yoda':
                    best_rect = [0, 0, 0, 0, 0]  # [x,y,w,h,area]
                    boxes = []
                    path_annotation = path_annotation + "/" + '%05d' % int(my_index1) + ".png"


                    #path_annotation = filename_target
                    print("path annotation : " + path_annotation)
                    img_annotation = cv2.imread(path_annotation)
                    copy_img_annotation = img_annotation.copy()
                    gray_mask_annotation = cv2.cvtColor(img_annotation, cv2.COLOR_RGB2GRAY)
                    thresh_annotation = cv2.threshold(gray_mask_annotation, soglia, 255, cv2.THRESH_BINARY)[1]
                    contours_annotation = cv2.findContours(thresh_annotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_annotation = contours_annotation[0] if len(contours_annotation) == 2 else contours_annotation[1]
                    boxes = []
                    for cntr in contours_annotation:
                        x, y, w, h = cv2.boundingRect(cntr)
                        #print("Bounding Box annotation {}".format(my_index1))
                        #print("x,y,w,h:", x, y, w, h)
                        boxes.append([x,y,w,h])
                    if len(boxes) != 0 :
                        best_x, best_y, best_w, best_h, best_area = max(boxes, key=lambda item: item[4])
                        print("Boxes :", boxes)
                        print("MAX :", best_x, best_y, best_w, best_h, best_area)
                        best_rect = [best_x, best_y, best_w, best_h, best_area]
                        for j in boxes:
                            if j != best_rect:
                                print(j)
                                if j[0] > best_x and j[0] < best_x + best_w and j[0] + j[2] > best_x and j[0] + j[
                                    2] < best_x + best_w and j[1] > best_y and j[1] < best_y + best_h and j[1] + j[
                                    3] > best_y and j[1] + j[3] < best_y + best_h:
                                    print("bbox compresa")
                                #else:
                                #    cv2.rectangle(copy_img_annotation, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]),(255, 0, 0), 2)
                                #    f_annotation.write(str(my_index) + "," + str(j[0]) + "," + str(j[1]) + "," + str(j[2]) + "," + str(j[3]) + "\n")
                            else:
                                # best box
                                cv2.rectangle(copy_img_annotation, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]),(255, 0, 0), 2)
                                f_annotation.write(str(my_index) + "," + str(j[0]) + "," + str(j[1]) + "," + str(j[2]) + "," + str(j[3]) + "\n")
                    else:
                        print("scrivo in annotazione 0,0,0,0")
                        f_annotation.write(str(my_index) + ",0,0,0,0" + "\n")


                    save_dir_bba = os.path.join(save_dir_res, "Bounding_box_annotations")
                    if not os.path.exists(save_dir_bba):
                        os.makedirs(save_dir_bba)
                    cv2.imwrite(os.path.join(save_dir_bba, 'BoundingBox_annotation_{}.png'.format(my_index)), copy_img_annotation)
                    f_annotation.close()

                else:
                    path_annotation = path_annotation + "/" + '%06d' % int(my_index1) + ".xml"
                    root = ET.parse(path_annotation).getroot()
                    l = []
                    for t in root:
                        l.append(t.tag)
                    if not l.__contains__("object"):
                        xmax = 0
                        xmin = 0
                        ymax = 0
                        ymin = 0
                    else:
                        childs = root[4]
                        child = childs[2]
                        xmax = child[0].text
                        xmin = child[1].text
                        ymax = child[2].text
                        ymin = child[3].text
                    f_annotation.write(str(my_index) + "," + str(xmin) + "," + str(ymin) + "," + str(int(xmax)-int(xmin)) + "," + str(int(ymax)-int(ymin)) + "\n")



                #draw BoundingBox on mask and on original img

                img_mask = cv2.imread(seg_filename)
                img_original = cv2.imread(filename_target)
                #img_original = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
                img_original = cv2.resize(img_original, (original_shape[1], original_shape[0]))
                result_mask = img_mask.copy()
                result_original = img_original.copy()
                #result_mask_full = img_mask.copy()
                #result_original_full = img_original.copy()
                gray_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY)
                thresh = cv2.threshold(gray_mask, soglia, 255, cv2.THRESH_BINARY)[1]
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                best_rect = [0,0,0,0,0]      # [x,y,w,h,area]
                boxes = []

                for cntr in contours:
                    x, y, w, h = cv2.boundingRect(cntr)
                    #print("Bounding Box img {}".format(my_index1))
                    #print("x,y,w,h:", x, y, w, h)
                    boxes.append([x,y,w,h,w*h])
                    #print("Boxes :", boxes)

                # File txt for save bbox detected

                #text_dir = os.path.join(save_dir_res, 'Txt')
                #if not os.path.exists(text_dir):
                #    os.makedirs(text_dir)

                box_text_filename = os.path.join(text_dir, 'boxes.txt')
                if os.path.exists(box_text_filename):
                    f = open(box_text_filename, "a")
                else:
                    f = open(box_text_filename, "w")

                if len(boxes) != 0 :
                    best_x, best_y, best_w, best_h, best_area = max(boxes, key=lambda item: item[4])
                    print("Boxes :", boxes)
                    print("MAX :" , best_x, best_y, best_w, best_h, best_area)
                    best_rect = [best_x,best_y,best_w,best_h,best_area]
                    for j in boxes:
                        if j != best_rect:
                            print(j)
                            if j[0]>best_x and j[0]<best_x+best_w and j[0]+j[2]>best_x and j[0]+j[2]<best_x+best_w and j[1]>best_y and j[1]<best_y+best_h and j[1]+j[3]>best_y and j[1]+j[3]<best_y+best_h:
                                print("bbox compresa")
                            else:
                                #cv2.rectangle(result_mask,(j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (0, 0, 255), 2)
                                cv2.rectangle(result_original,(j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (255, 0, 0), 2)
                                f.write(str(my_index) + ",0," + str(j[0]) + "," + str(j[1]) + "," + str(j[2]) + "," + str(j[3]) + "\n")
                        else:
                            # best box
                            #cv2.rectangle(result_mask, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (0, 0, 255), 2)
                            cv2.rectangle(result_original, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (255, 0, 0), 2)
                            #cv2.rectangle(result_mask_full, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (0, 0, 255), 2)
                            #cv2.rectangle(result_original_full, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (255, 0, 0), 2)
                            f.write(str(my_index)+",0,"+str(j[0])+","+str(j[1])+","+str(j[2])+","+str(j[3])+"\n")
                            #print("stringa che salvo nel file txt: [" + str(my_index)+",0,"+str(j[0])+","+str(j[1])+","+str(j[2])+","+str(j[3])+"]")

                    #save_dir_bbf = os.path.join(save_dir_res, "Bounding_box_full")
                    save_dir_bb = os.path.join(save_dir_res, "Bounding_box")
                    #save_dir_mf = os.path.join(save_dir_res, "Bounding_mask_full")
                    save_dir_m = os.path.join(save_dir_res, "Bounding_mask")
                    #if not os.path.exists(save_dir_bbf):
                    #    os.makedirs(save_dir_bbf)
                    if not os.path.exists(save_dir_bb):
                        os.makedirs(save_dir_bb)
                    #if not os.path.exists(save_dir_mf):
                    #    os.makedirs(save_dir_mf)
                    if not os.path.exists(save_dir_m):
                        os.makedirs(save_dir_m)

                    # save resulting image
                    #cv2.imwrite(os.path.join(save_dir_m, 'BoundingBox_mask_{}.png'.format(my_index1)), result_mask)
                    cv2.imwrite(os.path.join(save_dir_bb, 'BoundingBox_img_{}.png'.format(my_index1)), cv2.cvtColor(result_original, cv2.COLOR_RGB2BGR))
                    #cv2.imwrite(os.path.join(save_dir_mf, 'BoundingBox_mask_full_{}.png'.format(my_index1)), result_mask_full)
                    #cv2.imwrite(os.path.join(save_dir_bbf, 'BoundingBox_img_full_{}.png'.format(my_index1)), cv2.cvtColor(result_original_full, cv2.COLOR_RGB2BGR))

                f.close()



            else:
                print("dataset error")

        #'''

    if args.step == '2':

        if args.dataset == 'davis' or args.dataset == 'davis_yoda':
            f_val_seq = open("./val_seqs1.txt", "r")
            my_index = 0
        else:
            f_val_seq = open("./val_seqs2.txt", "r")
            my_index = -1
        img_sequencies_name = [x.strip() for x in f_val_seq.readlines()]
        print(img_sequencies_name)
        f_val_seq.close()


        acc = mm.MOTAccumulator(auto_id=True)
        # Avvio tracker
        if args.dataset == 'davis' or args.dataset == 'davis_yoda':
            path_original_img = os.path.join(args.data_dir, "JPEGImages/480p")
        else:
            path_original_img = os.path.join(args.data_dir, 'Data/VID/val')
        print("Avvio tracker su " + path_original_img)
        path_boxes_txt = os.path.join(args.seg_save_dir, 'Results_{}'.format(soglia))
        box_tracker.main(img_sequencies_name, path_original_img, path_boxes_txt)

        # Ho ottenuto tutte le bbox da prendere in considerzione per tutte le img
        my_index = 0
        old_temp = ''


        print("INIZIO SECONDO STEP")
        # STEP 2, aggiorno frame per frame l'accumulatore
        for index, batch in enumerate(testloader):

            temp = batch['seq_name']
            #args.seq_name = temp[0]

            if index == 0 and args.data_dir == '/mnt/ILSVRC2017_VID/ILSVRC':
                args.seq_name = img_sequencies_name[0]

            if (args.data_dir == '/data/aacunzo/DAVIS-2016' or args.data_dir == '/home/aacunzo/DAVIS-2016'):
                args.seq_name = temp[0]
                print("old_temp: " , old_temp)
                if old_temp==args.seq_name:
                    my_index = my_index+1
                else:
                    my_index = 0

            if (args.data_dir == '/mnt/ILSVRC2017_VID/ILSVRC'):
                path_original_img = os.path.join(args.data_dir, 'Data/VID/val')
                path_original_img = path_original_img + "/" + args.seq_name
                end = len([name for name in os.listdir(path_original_img) if os.path.isfile(os.path.join(path_original_img, name))])
                print("num img :", end)
                print("old_temp: ", old_temp)
                if my_index == end :
                    my_index = 0
                    cont = cont+1
                    args.seq_name = img_sequencies_name[cont]
                else:
                    my_index = my_index + 1

            print("my_index : ",my_index)
            print("cont : ",cont)
            print("seq name : ",args.seq_name)

            if my_index==0:
                print("Primo frame della sequenza " , args.seq_name)
                save_dir_res = os.path.join(args.seg_save_dir, 'Results_{}'.format(soglia), args.seq_name)
                text_dir = os.path.join(save_dir_res, 'Txt')
                save_dir_res_final =  text_dir
                if args.mode == 'good':
                    box_text_filename = os.path.join(text_dir, 'boxes_good.txt')
                else:
                    box_text_filename = os.path.join(text_dir, 'boxes.txt')

                box_text_annotation_filename = os.path.join(text_dir, 'boxes_annotations.txt')

                f = open(box_text_filename, "r")
                f_annotation = open(box_text_annotation_filename, "r")

                all_annotations = []
                all_annotations = [x.strip() for x in f_annotation.readlines()]
                all_annotations = [x.split(',') for x in all_annotations]

                first = all_annotations[0]
                old = first[0]
                old_area = first[3]*first[4]
                c = 0
                for i in all_annotations:
                    if i[0] == old & c!=0:
                        #non è il primo elemento
                        area = i[3] * i[4]
                        if area > old_area:
                            all_annotations.remove(all_annotations.index(i)-1)
                        else:
                            all_annotations.remove(i)
                            print("Rimosso :")
                            print(i)
                    old = i[0]
                    c=1
                print("frame video secondo all_annotations modificato :", len(all_annotations))

                #i.remove(i[0])

                all_boxes = [x.strip() for x in f.readlines()]
                all_boxes = [x.split(',') for x in all_boxes]
                #value_correct = int(all_boxes[0][0])
                #print(value_correct)


                f.close()

                f_annotation.close()

            hypotheses = []
            box_in_frame = []
            distances = []
            objs = []
            hyps = []

            for i in all_boxes:
                #if int(i[0])-value_correct == my_index:
                if int(i[0]) == my_index:
                    #print("i in all_boxes")
                    #print(i)
                    if(i.__len__()==6):
                        box_in_frame.append(i)

            #print(box_in_frame)

            for z in box_in_frame:
                z.remove(z[1])
                z.remove(z[0])

            #print(box_in_frame)

            print("box nel frame " + str(my_index) + " : ", box_in_frame)

            for j in range(0, len(box_in_frame)):
                #print(box_in_frame[j])
                hypotheses.append(j+1)

            print("ipotesi nel frame " + str(my_index) + " : ", hypotheses)

            objs = all_annotations[my_index]
            if not objs == "0,0,0,0" :
                #print(objs)
                objs = np.array(objs)
                #print(objs.shape)
                # aggiungere asse objs
                objs = np.expand_dims(objs, 0)
                #print(objs.shape)
                # objs.shape - ---> (4,)
                # objs.shape - ---> (1, 4)
                if box_in_frame.__len__() == 0:
                    box_in_frame.append('0,0,0,0')
                for a in box_in_frame:
                    #print(a)
                    hyps.append(a)
                print(hyps)
                hyps = np.array(hyps)
                #hyps = np.expand_dims(hyps, 0)
                print("Compute IOU")
                print("Objects : ", objs)
                print("Num hypothesis :" , hypotheses)
                print("Hypothesis : ", hyps)

                distances = mm.distances.iou_matrix(objs, hyps, max_iou=0.5)
                print(distances)
                print("---------------------------")

                acc.update(
                    [1],  # Ground truth objects in this frame
                    hypotheses,  # Detector hypotheses in this frame
                    [
                        distances,  # Distances from object 1 to hypotheses 1, 2, 3
                    ]
                )


            old_temp = args.seq_name

        text_dir = os.path.join(path_boxes_txt, 'TEST')
        print(text_dir)
        if not os.path.exists(text_dir):
            os.makedirs(text_dir)



        if args.mode == 'good':
            if args.type == '1':
                results_filename = os.path.join(save_dir_res_final,'results_good_' + str(img_sequencies_name.__len__()) + '-' + str(string_data) + '.txt')
            else:
                results_filename = os.path.join(text_dir, 'results_good_'+ str(img_sequencies_name.__len__()) + '-' + str(string_data) + '.txt')
        else:
            if args.type == '1':
                results_filename = os.path.join(save_dir_res_final,'results_good_' + str(img_sequencies_name.__len__()) + '-' + str(string_data) + '.txt')
            else:
                results_filename = os.path.join(text_dir, 'results_'+ str(img_sequencies_name.__len__())+ '-' + str(string_data) + '.txt')

        f_results = open(results_filename,'w')

        #f_results.write("\n ACC EVENTS \n")
        #f_results.write(str(acc.events))
        f_results.write("\n ACC MOT EVENTS \n")
        f_results.write(str(acc.mot_events))

        print(acc.events)
        print(acc.mot_events)

        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
        print(summary)
        f_results.write("\n ACC SUMMARY 1\n")
        f_results.write(str(summary))

        summary = mh.compute_many(
            [acc, acc.events.loc[0:1]],
            metrics=['num_frames', 'mota', 'motp'],
            names=['full', 'part'])
        print(summary)
        f_results.write("\n ACC SUMMARY 2\n")
        f_results.write(str(summary))

        '''
        strsummary = mm.io.render_summary(
            summary,
            formatters={'mota': '{:.2%}'.format},
            namemap={'mota': 'MOTA', 'motp': 'MOTP'}
        )
        print(strsummary)
        f_results.write("\n ACC SUMMARY \n")
        f_results.write(str(summary))
        

        summary = mh.compute_many(
            [acc, acc.events.loc[0:1]],
            metrics=mm.metrics.motchallenge_metrics,
            names=['full', 'part'])

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)
        f_results.write("\n ACC SUMMARY \n")
        f_results.write(str(summary))
        '''

        summary = mh.compute_many(
            [acc, acc.events.loc[0:1]],
            metrics=mm.metrics.motchallenge_metrics,
            names=['full', 'part'],
            generate_overall=True
        )

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)
        f_results.write("\n ACC SUMMARY FINAL\n")
        f_results.write(str(strsummary))
    
    #'''

    '''
    string_data = "30-6-2021-20-21-14"
    text_dir = "./result/test/davis_iteration_conf/Results_200/blackswan/Txt"
    box_text_filename = os.path.join(text_dir, 'boxes_' + string_data + '.txt')
    box_annotation_text_filename = os.path.join(text_dir, 'boxes_annotations_' + string_data + '.txt')
    path_boxes_txt = os.path.join(args.seg_save_dir, 'Results_{}'.format(soglia))
    box_tracker.main(img_sequencies_name,path_original_img,path_boxes_txt,string_data)

    f = open(box_text_filename, "r")
    f_annotation = open(box_annotation_text_filename, "r")
    all_annotations = [x.strip() for x in f_annotation.readlines()]
    all_annotations = [x.split(',') for x in all_annotations]

    for i in all_annotations:
        i.remove(i[0])

    #print(all_annotations)
    all_boxes = [x.strip() for x in f.readlines()]
    all_boxes = [x.split(',') for x in all_boxes]
    #print(all_boxes)
    #print(all_boxes[0]) # bbox specific
    #print(all_boxes[0][0]) # primo el bbox

    my_index = 0
    old_temp = ''

    hypotheses = []
    box_in_frame = []
    distances = []

    acc = mm.MOTAccumulator(auto_id=True)


    # Scorro ogni frame
    for index, batch in enumerate(testloader):

        temp = batch['seq_name']
        args.seq_name = temp[0]
        if old_temp == args.seq_name:
            my_index = my_index + 1
        else:
            my_index = 0

        hypotheses = []
        box_in_frame = []
        distances = []

        for i in all_boxes:
            if int(i[0]) == my_index:
                box_in_frame.append(i)

        for z in box_in_frame:
            z.remove(z[1])
            z.remove(z[0])

        print("box nel frame " + str(my_index) + " : " , box_in_frame)

        for j in range(1,len(box_in_frame)+1):
            hypotheses.append(j)

        print("ipotesi nel frame " + str(my_index) + " : ", hypotheses)


        objs = all_annotations[my_index]
        #aggiungere asse objs
        objs = np.expand_dims(objs, 0)
        objs.shape ----> (4, )
        objs.shape ----> (1,4)
        hyps = np.array(hypotheses)
        print("Compute IOU")
        print("Objects : ", objs)
        print("Hypothesis : ", hyps)

   
        #distances = mm.distances.iou_matrix(objs, hyps, max_iou=0.5)
        
        #acc.update(
        #[1],  # Ground truth objects in this frame
        #hypotheses,  # Detector hypotheses in this frame
        #[
        #    distances,  # Distances from object 1 to hypotheses 1, 2, 3
        #]
        #)
      
        acc.update(
            [1],  # Ground truth objects in this frame
            [1,2],  # Detector hypotheses in this frame
            [
                [0.7,0.3]  # Distances from object 1 to hypotheses 1, 2, 3
            ]
        )



        old_temp = args.seq_name

    print(acc.events)

    print(acc.mot_events)

    f.close()
    f_annotation.close()
    
    '''

    

if __name__ == '__main__':
    main()
