from PIL import Image
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, rawroot, maskroot, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[3]))#把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.maskroot = maskroot
        self.rawroot = rawroot
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.hog = cv2.HOGDescriptor((224,224), (16,16), (8,8), (8,8),9)


    def __getitem__(self, index):
        fn = self.imgs[index]
        rawimg = cv2.imread(self.rawroot + fn,cv2.IMREAD_GRAYSCALE)  # 按照path读入图片from PIL import Image # 按照路径读取图片
        #rawimg = cv2.resize(rawimg,(320,240))
        print(fn)
        hogfeature = self.hog.compute(rawimg, (8, 8),(0,0))
        if self.transform is not None:
            hogfeature = self.transform(hogfeature)  # 是否进行transformsform
        return hogfeature.transpose()  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):
        return len(self.imgs)
