from __future__ import print_function
import argparse
import os
from model import LF_Net
from util import is_image_file, load_img1, save_img
from loss import *


parser = argparse.ArgumentParser(description='JAAL-PyTorch-implementation')
parser.add_argument('--model', type=str,default='./checkpoint/netG_model_epoch_50.pth',help='model file to use')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=3, help='generator filters in first conv layer')
parser.add_argument('--cuda', action='store_true',default=True, help='use cuda')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
opt = parser.parse_args()
print(opt)

netG_state_dict = torch.load(opt.model)
netG = LF_Net(opt.input_nc, opt.output_nc)
netG.load_state_dict(netG_state_dict, strict=False)

image_dir = "./plant png/a/"
label_dir = "./plant png/b/"

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
label_filenames = [x for x in os.listdir(label_dir) if is_image_file(x)]
batchsize=2


with torch.no_grad():
    dt_size = 1
    acc=0
    dice=0
    iou=0
    sen=0
    spe=0
    i=0   
    for image_name,label_name in zip(image_filenames,label_filenames):
        img=load_img1(image_dir+image_name)
        label,shape=load_img1(label_dir+label_name)
        print(image_name,label_name)
        i=i+1
        print(i)
        input_x_np = np.zeros((batchsize, 3, 128, 128)).astype(np.float32)
        input_x_np[0,:] = np.asarray(img[0])
        input= Variable(torch.from_numpy(input_x_np))
            
        if opt.cuda:
            netG = netG.cuda()
            input = input.cuda()
    
        out = netG(input)
        out = out.cpu()
        out_img = out.data[0]

        save_img(out_img, "./plant png/c".format(label_name),shape)
        dice+=Dice(out_img,label)
        iou+=iou_score(out_img,label)
        
        sen+=get_sensitivity(out_img,label)
        spe+=get_specificity(out_img,label)
        acc+=get_accuracy(out_img,label)
        print("dice:%0.4f iou:%0.4f spe:%0.4f sen:%0.4f acc:%0.4f" % (dice/dt_size,iou/dt_size,spe/dt_size,sen/dt_size,acc/dt_size))

        
    


        
   

