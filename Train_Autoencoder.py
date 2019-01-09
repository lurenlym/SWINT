import argparse
import MyData_Autoencoder
import torch
import model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import numpy as np
import time
from torch.autograd import Variable
import torch.nn.functional as F
parser = argparse.ArgumentParser(description='DenseNet')
parser.add_argument('--save', type=str, default='./DenseNet.pt',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--unuse-cuda', action='store_true',
                    help='unuse cuda')

parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs for train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size for training')
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--dropout', type=float, default=0.)

parser.add_argument('--num-class', type=int, default=2)
parser.add_argument('--growth-rate', type=int, default=12)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--rawpicroot', type=str, default='C:\\Users\\lyming\\Desktop\\SFCdata\\raw\\')
parser.add_argument('--maskpicroot', type=str, default='C:\\Users\\lyming\\Desktop\\SFCdata\\mask\\maskdata\\')
parser.add_argument('--Train', type=bool, default=False)
args = parser.parse_args()

#transform=transforms.Compose([transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    #transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
#])


use_cuda = torch.cuda.is_available() and not args.unuse_cuda
args.dropout = args.dropout if args.augmentation else 0.

train_data=MyData_Autoencoder.MyDataset(maskroot=args.maskpicroot,rawroot=args.rawpicroot,datatxt='train.txt', transform=None)
data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
#test_data=MyData_Autoencoder.MyDataset(maskroot=args.maskpicroot,rawroot=args.rawpicroot,datatxt='test.txt', transform=None)
#test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
print(len(data_loader))

# ##############################################################################
# Build model
# ##############################################################################



autoencoder = model.AutoEncoder(args)
if args.Train==True:
    autoencoder._initialize_weights()
else:
    autoencoder.load_state_dict(torch.load(args.save)['model'])
if use_cuda:
    SFCNET = autoencoder.cuda()
optimizer = torch.optim.Adam(SFCNET.parameters(), lr=args.lr)

criterion =torch.nn.MSELoss()
testtransform =transforms.ToTensor()
# ##############################################################################
# Training
# ##############################################################################



def train():
    corrects = total_loss = 0
    for i, (rawdata) in enumerate(data_loader):
        rawdata = Variable(rawdata)
        if use_cuda:
            rawdata = rawdata.cuda()

        OUTPUT = autoencoder(rawdata)
        loss = criterion(OUTPUT, rawdata)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]
        #corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()

    return total_loss

def test():
    fh = open('test.txt', 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
    imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
    for line in fh:  # 按行循环txt文本中的内容
        line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
        words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
        imgs.append((words[0]))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
    hog = cv2.HOGDescriptor((320, 240), (16, 16), (8, 8), (8, 8), 9)

    for i in range(len(imgs)):
        fn = imgs[i]
        rawimg = cv2.imread(args.rawpicroot + fn, cv2.IMREAD_GRAYSCALE)  # 按照path读入图片from PIL import Image # 按照路径读取图片
        rawimg = cv2.resize(rawimg, (320, 240))
        hogfeature = hog.compute(rawimg, (8, 8), (0, 0))
        hogfeature = hogfeature.transpose()
        rawdata = Variable(torch.from_numpy(hogfeature))
        if use_cuda:
            rawdata = rawdata.cuda()

        OUTPUT = autoencoder(rawdata)
        loss = criterion(OUTPUT, rawdata)
        print(fn,loss[0])
    return loss[0]
# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    if args.Train==True:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            loss= train()
            print(epoch,loss)
            #optimizer.update_learning_rate()
    else:
            result = test()
            print(result)

    model_state_dict = SFCNET.state_dict()
    model_source = {
        "settings": args,
        "model": model_state_dict
    }
    torch.save(model_source, args.save)
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

