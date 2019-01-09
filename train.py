import argparse
import MyData
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

parser.add_argument('--lr', type=float, default=0.000001)
parser.add_argument('--epochs', type=int, default=500,
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

transform=transforms.Compose([
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
])


use_cuda = torch.cuda.is_available() and not args.unuse_cuda
args.dropout = args.dropout if args.augmentation else 0.

train_data=MyData.MyDataset(maskroot=args.maskpicroot,rawroot=args.rawpicroot,datatxt='lists.txt', transform=transform)
data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
print(len(data_loader))

# ##############################################################################
# Build model
# ##############################################################################



SFCNET = model.SFCNET(args)
if args.Train==True:
    SFCNET._initialize_weights()
else:
    SFCNET.load_state_dict(torch.load(args.save)['model'])
if use_cuda:
    SFCNET = SFCNET.cuda()
optimizer = torch.optim.Adadelta(SFCNET.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
L1loss = torch.nn.L1Loss()
testtransform =transforms.ToTensor()
# ##############################################################################
# Training
# ##############################################################################



def train():
    corrects = total_loss = 0
    for i, (rawdata, maskdata, label) in enumerate(data_loader):
        maskdata, label = Variable(maskdata), Variable(label)
        rawdata = Variable(rawdata)
        if use_cuda:
            rawdata, maskdata, label = rawdata.cuda(),maskdata.cuda(), label.cuda()

        QI = SFCNET(rawdata)
        QIm = SFCNET(maskdata)
        QI = QI.view(args.batch_size,2).detach()
        QIm = QIm.view(args.batch_size, 2)
        #label = label.view(2)
        losscla = criterion(QIm, label)
        lossdis = L1loss(QIm, QI)
        loss = losscla+0.1*lossdis
        #loss = losscla
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        #corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()

    return total_loss[0]

def sliding_window(image, padimg,stepSize, windowSize):
	# slide a window across the image
	for y in range(windowSize[0], padimg.shape[0]-windowSize[0], stepSize):
		for x in range(windowSize[1], padimg.shape[1]-windowSize[1], stepSize):
		    yield (x-windowSize[1], y-windowSize[0], padimg[y - int(windowSize[0]/2):y + int(windowSize[0]/2), x- int(windowSize[1]/2):x + int(windowSize[1]/2)])

def test():
    testimg = cv2.imread("C:\\Users\lyming\Desktop\SFCdata\mask\maskdata\\28.jpg")
    testimgshape = np.shape(testimg)
    testsize = (400, 200)
    padimg = np.pad(testimg,((400, 400),(200, 200),(0, 0)),mode = 'wrap')
    result = np.zeros((testimgshape))
    for [x,y,testdata] in sliding_window(testimg,padimg,2,testsize):
        testdata = cv2.resize(testdata, (224, 224))
        testdata = Variable(testtransform(testdata))
        testdata = testdata.cuda()
        testdata = testdata.unsqueeze(0)
        testdata = SFCNET(testdata)
        testdata = F.softmax(testdata.view(2,),dim=0)
        result[y,x,0:2] = testdata.detach().cpu().numpy()
        print(x, y, testdata.shape,result[y,x,0:2])
    return result[:,:,0:2]
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
        Pmap = result[:, :, 0]
        Nmap = result[:, :, 1]
        Pmap = cv2.GaussianBlur(Pmap, (11, 11), 15)*255
        Nmap = cv2.GaussianBlur(Nmap, (11, 11), 15)*255
        cv2.imwrite("Nmap.jpg",Nmap)
        cv2.imwrite("Pmap.jpg", Pmap)
    model_state_dict = SFCNET.state_dict()
    model_source = {
        "settings": args,
        "model": model_state_dict
    }
    torch.save(model_source, args.save)
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

