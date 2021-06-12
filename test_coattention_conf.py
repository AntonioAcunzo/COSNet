import argparse
import torch
import torch.nn as nn
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
        args.data_dir = '/thecube/students/lpisaneschi/ILSVRC2017_VID/ILSVRC'   # 37572 image pairs
        args.data_list = '/thecube/students/lpisaneschi/ILSVRC2017_VID/ILSVRC/val_seqs1.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '473,473' #Comma-separated string with height and width of images
        args.num_classes = 2      #Number of classes to predict (including background) ****
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        args.restore_from = './co_attention.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = './snapshots/imagenet_iteration/'          #Where to save snapshots of the model
        args.save_segimage = True
        args.seg_save_dir = "./result/test/imagenet_iteration_conf"
        args.vis_save_dir = "./result/test/imagenet_vis"
        args.corp_size =(473, 473)
        
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

    if args.dataset == 'voc12':
        testloader = data.DataLoader(VOCDataTestSet(args.data_dir, args.data_list, crop_size=(505, 505),mean= args.img_mean), 
                                    batch_size=1, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=(505, 505), mode='bilinear')
        voc_colorize = VOCColorize()

    if args.dataset == 'imagenet':  #for imagenet
        db_test = db.PairwiseImg(train=False, inputRes=(473,473), db_root_dir=args.data_dir,  transform=None, seq_name = None, sample_range = args.sample_range) #db_root_dir() --> '/path/to/DAVIS-2016' train path
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
    my_index = 0
    old_temp=''


    for index, batch in enumerate(testloader):
        print("----------------------------------------------------------------------------------------------------------------------")
        print("processed index: ", '%d processed'%(index))
        target = batch['target']
        #search = batch['search']
        temp = batch['seq_name']

        args.seq_name=temp[0]

        print("Target : ", target) # Tensor
        print("Target shape: ", target.shape) # torch.Size([1, 3, 473, 473])
        print("Temp : ", temp) # [blackswan]
        print("Seq_name : ", args.seq_name) # blackswan

        print("Target max value : ", torch.max(target))
        print("Target min value : ", torch.min(target))

        path_save_img = "./IMG_PROVA"
        filename = os.path.join(path_save_img, 'target_t.png')

        img_target = target[0] # torch.Size([3, 473, 473])
        print("img target: ", img_target)

        PIL_img = transforms.ToPILImage()(img_target)
        #PIL_img.convert("RGB")
        PIL_img.save(filename)

        img1 = Image.open(os.path.join(path_save_img, 'target.png'))

        #-----------------------------

        img_target_R = img_target[0]
        img_target_G = img_target[1]
        img_target_B = img_target[2]

        print("img target R: ", img_target_R)
        print("img target G: ", img_target_G)
        print("img target B: ", img_target_B)

        print(db_test.meanval[0])
        targe_R = img_target_R.numpy() + db_test.meanval[0]
        print("target R : ", targe_R)

        #-------------------------------

        img_target_numpy = img_target.numpy()
        print("img target numpy: ", img_target_numpy)
        img_target_numpy = img_target_numpy.transpose((1, 2, 0))  # CHW --> HWC
        print("img target numpy after transpose: ", img_target_numpy)

        img_target_numpy = img_target_numpy + np.array(db_test.meanval)
        print("img target numpy denorm: ", img_target_numpy)

        img_target_numpy = Image.fromarray(img_target_numpy.astype(np.uint8))

        PIL_img_from_numpy = Image.fromarray(img_target_numpy)
        filename = os.path.join(path_save_img, 'target_n.png')
        PIL_img.save(filename)

        #-----------------------------------

        #x1 = img_target

        #z1 = x1 * torch.tensor(torch.std(x1)).view(3, 1, 1)
        #z1 = z1 + torch.tensor(torch.mean(x1)).view(3, 1, 1)

        #img2 = transforms.ToPILImage(mode='RGB')(z1)
        #filename2 = os.path.join(path_save_img, 'target2.png')
        #img2.save(filename2)


        #img_target = img_target.numpy()[:,:,:]
        #torch.squeeze(img_target,0)
        print("Img target shape: ", img_target.shape)  # torch.Size([3, 473, 473])

        #img_target = np.repeat(img_target.numpy()[None,:,:],3,axis=-1)
        #print("Img target shape: ", img_target.shape)

        #img_target = img_target.numpy()[:, :, :]
        #img_target = np.squeeze(img_target, axis=0)
        #print("Img target shape: ", img_target.shape)

        #print("max value in target : ", np.max(img_target))  #


        #img1 = Image.fromarray((img_target * 255).astype(np.uint8))
        #img1.save(filename)


        #save_image(img1,filename)
        #imgs = Image.fromarray(imgs)
        #imgs = imgs.convert("L")
        #imgs.save(filename)



        if old_temp==args.seq_name:
            my_index = my_index+1
        else:
            my_index = 0

        print("my_index : ",my_index) # 0,1,2...

        output_sum = 0

        print("Sample_range : ", args.sample_range) # 5

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
            '''
            path = "./IMG_PROVA"
            my_index2 = str(i).zfill(5)
            filename = os.path.join(path, 'search_{}.png'.format(my_index2))
            print(filename)
            img = Image.fromarray(target)
            img = img.convert("L")
            img.save(filename)
            '''
        
        output1 = output_sum/args.sample_range

        #print("Output1 : ", output1)
        print("max value in output1 : ",np.max(output1)) # 0.99999
        print("Output1 shape: ", output1.shape) # (473, 473)

        if(args.data_dir == '/data/aacunzo/DAVIS-2016' or args.data_dir == '/home/aacunzo/DAVIS-2016'):
            first_image = np.array(Image.open(args.data_dir+'/JPEGImages/480p/blackswan/00000.jpg'))
        if (args.data_dir == '/thecube/students/lpisaneschi/ILSVRC2017_VID/ILSVRC'):
            first_image = np.array(Image.open(args.data_dir + '/Data/VID/val/ILSVRC2015_val_00000000/000000.JPEG'))

        original_shape = first_image.shape
        print("Original shape :", original_shape) # (480, 854, 3)
        output1 = cv2.resize(output1, (original_shape[1],original_shape[0]))
        print("Output1 shape after resize : ", output1.shape) # (480, 854)

        print("output1 stampa : ", output1)
        mask = (output1*255).astype(np.uint8)
        print("MASK stampa: ", mask)
        #mask_array = mask
        print("mask size :", mask.shape)
        mask = Image.fromarray(mask)

        '''
        path = "./IMG_PROVA"
        my_index2 = str(my_index).zfill(5)
        filename = os.path.join(path, 'target_{}.png'.format(my_index2))
        print(filename)
        target1 = target.numpy()[:, :, :, :]
        img = Image.fromarray(target1)
        img = img.convert("L")
        img.save(filename)
        '''


        if args.dataset == 'voc12':
            print(output.shape)
            print(size)
            output = output[:,:size[0],:size[1]]
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            if args.save_segimage:
                seg_filename = os.path.join(args.seg_save_dir, '{}.png'.format(name[0]))
                color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
                color_file.save(seg_filename)

        if args.dataset == 'imagenet':

            save_dir_res = os.path.join(args.seg_save_dir, 'Results-Imagenet', args.seq_name)
            old_temp = args.seq_name
            if not os.path.exists(save_dir_res):
                os.makedirs(save_dir_res)
            if args.save_segimage:
                my_index1 = str(my_index).zfill(5)
                seg_filename = os.path.join(save_dir_res, '{}.png'.format(my_index1))
                #color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
                mask.save(seg_filename)
                #np.concatenate((torch.zeros(1, 473, 473), mask, torch.zeros(1, 512, 512)),axis = 0)
                #save_image(output1 * 0.8 + target.data, args.vis_save_dir, normalize=True)
                
        elif args.dataset == 'davis' or args.dataset == 'davis_yoda':
            
            save_dir_res = os.path.join(args.seg_save_dir, 'Results', args.seq_name)
            old_temp=args.seq_name
            if not os.path.exists(save_dir_res):
                os.makedirs(save_dir_res)
            if args.save_segimage:   
                my_index1 = str(my_index).zfill(5)
                seg_filename = os.path.join(save_dir_res, '{}.png'.format(my_index1))
                #color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
                mask.save(seg_filename)

                '''
                mask_array = mask_array[np.newaxis ,:, :]
                print("Mask_a shape : ",mask_array.shape)

                #mask_img = Image.fromarray(mask_array)

                a = np.concatenate((torch.zeros(1, 473, 473), mask_array, torch.zeros(1, 512, 512)), axis=0)
                a = Image.fromarray(a)
                a.save("./IMG_PROVA/prova.png")

                #np.concatenate((torch.zeros(1, 473, 473), mask, torch.zeros(1, 512, 512)),axis = 0)
                #save_image(output1 * 0.8 + target.data, "./IMG_PROVA/prova.png", normalize=True)
                '''
        else:
            print("dataset error")

        break
            


    

if __name__ == '__main__':
    main()
