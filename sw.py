import cv2
import numpy as np

def sliding_window(image, padimg,stepSize, windowSize):
	# slide a window across the image
	for y in range(windowSize[0], padimg.shape[0]-windowSize[0], stepSize):
		for x in range(windowSize[1], padimg.shape[1]-windowSize[1], stepSize):
		    yield (x - windowSize[1], y-windowSize[0], padimg[y - int(windowSize[0]/2):y + int(windowSize[0]/2), x- int(windowSize[1]/2):x + int(windowSize[1]/2)])


fh = open('wuhanlists.txt', 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
for line in fh:  # 按行循环txt文本中的内容
    line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
    words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
    imgs.append((words[0]))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
    # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
rawroot = "C:\\Users\\lyming\\Desktop\\SFCdata\\raw\\"

for index in range(len(imgs)):
    fn = imgs[index]
    rawimg = cv2.imread(rawroot + fn,cv2.IMREAD_GRAYSCALE)
    for [x, y, testdata] in sliding_window(rawimg, rawimg, 20, (224,224)):
        cv2.imwrite("C:\\Users\lyming\Desktop\SFCdata\\re\\"+str(x)+"_"+str(y)+fn, testdata)
        #print(fn)# 是否进行transformsform