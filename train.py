from __future__ import print_function
import argparse
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import LF_Net,DA_Net
from data import get_training_set, get_validate_set
import torch.backends.cudnn as cudnn
from adjust import train_split_validate
from loss import *


parser = argparse.ArgumentParser(description='JAAL-PyTorch-implementation')
parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=2, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True,help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
opt = parser.parse_args()
print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
cudnn.benchmark = True
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    

print('===> Loading datasets')
root_path = "../dataset/"
train_split_validate()
train_set = get_training_set(root_path)
validate_set = get_validate_set(root_path)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
validate_data_loader = DataLoader(dataset=validate_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

print("train set length",len(train_set))
print("validate set:",len(validate_set))
print('===> Building model')
netG =  LF_Net(opt.input_nc, opt.output_nc)
pytorch_total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
print ("\nTrainable parameters", pytorch_total_params)
netD = DA_Net(opt.input_nc, opt.output_nc, opt.ndf)

criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion_mse = nn.MSELoss()

real_A = torch.FloatTensor(opt.batchSize, opt.input_nc, 128, 128)
real_B = torch.FloatTensor(opt.batchSize, opt.output_nc, 128, 128)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterion = criterion.cuda()
    criterion_l1 = criterion_l1.cuda()
    criterion_mse = criterion_mse.cuda()
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    label = label.cuda()

real_A = Variable(real_A)
real_B = Variable(real_B)
label = Variable(label)

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):
        netD.volatile = False
        netD.zero_grad()
        real_a_cpu, real_b_cpu = batch[0], batch[1]
    
        real_A.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_B.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)

        output = netD(torch.cat((real_A, real_B), 1))
        label.resize_(output.size()).fill_(real_label)

        err_d_real = criterion(output, label)
        err_d_real.backward()
        d_x_y = output.data.mean()
        

        fake_b = netG(real_A)
        output = netD(torch.cat((real_A, fake_b.detach()), 1))
        label.data.resize_(output.size()).fill_(fake_label)

        err_d_fake = criterion(output, label)
        err_d_fake.backward()
        d_x_gx = output.data.mean()
        err_d = (err_d_real + err_d_fake) / 2.0
        optimizerD.step()

        output = netD(torch.cat((real_B, fake_b.detach()), 1))
        label.data.resize_(output.size()).fill_(fake_label)

        err_d_fake = criterion(output, label)
        err_d_fake.backward()
        d_x_gx = output.data.mean()
        err_d = (err_d_real + err_d_fake) / 2.0
        optimizerD.step()

        netG.zero_grad()
        netD.volatile = True
        output = netD(torch.cat((real_A, fake_b), 1))
        
        label.data.resize_(output.size()).fill_(real_label)

        err_g = criterion(output, label) + opt.lamb * criterion_l1(fake_b, real_B)+50*EPE(fake_b, real_B)
        err_g.backward()
        d_x_gx_2 = output.data.mean()
        optimizerG.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f}  Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}".format(
            epoch, iteration, len(training_data_loader), err_d.item(), err_g.item(), d_x_y, d_x_gx, d_x_gx_2))



def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    net_g_model_out_path = "../result/checkpoint/netG_model_epoch_{}.pth".format(epoch)
    net_d_model_out_path = "../result/checkpoint/netD_model_epoch_{}.pth".format(epoch)
    torch.save(netG.state_dict(), net_g_model_out_path)
    torch.save(netD.state_dict(), net_d_model_out_path)
    print("Checkpoint saved to ".format("checkpoint"))


def validate(epoch):
    dt_size = 50.0/2
    acc=0
    dice=0
    iou=0
    sen=0
    spe=0
    i=0
    for batch in validate_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            
        netG_state_dict = torch.load("../result/checkpoint/netG_model_epoch_{}.pth".format(epoch))
        netG.load_state_dict(netG_state_dict)

        prediction = netG(input)
        prediction=prediction.data[0]
        target=target.data[0]
        i=i+1
        print(i)
        dice+=Dice(prediction,target)
        iou+=iou_score(prediction,target)
        sen+=get_sensitivity(prediction,target)
        spe+=get_specificity(prediction,target)
        acc+=get_accuracy(prediction,target)
        print("dice:%0.3f iou:%0.3f spe:%0.3f sen:%0.3f acc:%0.3f" % (dice/dt_size,iou/dt_size,spe/dt_size,sen/dt_size,acc/dt_size))

        

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    if epoch % 10==0:
        checkpoint(epoch)
        validate(epoch)
    

