#@Luffy 2018年2月13日创建
#将MNIST数据集的train图片及lable分割成 number_index.bmp的形式。
#number是图片的标签，即0~9十个数字
#index是当前图片在图片所在数字中的序号
#图片的类型是bmp，未压缩全彩。当然，MNIST的图片是黑底灰度图像。
import struct
import numpy as np
from PIL import Image

#打开image文件,自行下载MNIST数据集，数据集包含4个文件，自行解压，并放置在程序所在目录下的 MNIST_data目录下
#并新建/MNIST_data/train/目录，用于保存输出的数字图片。
filename_image='MNIST_data/train-images.idx3-ubyte'
bin_image=open(filename_image,'rb')
buf_image=bin_image.read()
#打开label文件
filename_label='MNIST_data/train-labels.idx1-ubyte'
bin_label=open(filename_label,'rb')
buf_label=bin_label.read()
#struct使用自行百度参考。本案例使用大端法
#设定IMAGE文件流的位移指针
index_image=0
#一个I是4个字节的int，前4个字节的int分别表示魔数、本文件所含图片的数量、单个图片的rows，单个图片的cols
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf_image,index_image)
#图片从第16个字节开始，0为文件的起始
index_image+=struct.calcsize('>IIII')
print(magic,numImages,index_image)
#设定label文件流指针
index_label=0
#label文件前两个int是魔数、label的总数
magic,numlabels = struct.unpack_from('>II',buf_label,index_label)
index_label+=struct.calcsize('>II')
print(magic,numlabels,index_label)
#定义包含10个数字的数组，保存当前图片的index
#例如图片的名称是0_1042.bmp、9_345.bmp,1042，345就是index
inum=[0,0,0,0,0,0,0,0,0,0]

for i in range(0,numImages):
    im=struct.unpack_from('>784B',buf_image,index_image)
    index_image+=struct.calcsize('>784B')
    #f返回的lb是一个tuple类型，tuple实际上是个数组
    lb=struct.unpack_from('>1B',buf_label,index_label)
    index_label+=struct.calcsize('>1B')
    
    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)
    im=Image.fromarray(im)
    #文件名的形态 number_index.bmp
    im.save('MNIST_data/train/%s_%s.bmp'%(lb[0],inum[lb[0]]),'bmp')
    print('MNIST_data/train/%s_%s.bmp'%(lb[0],inum[lb[0]]))
    #数字的index向前进1
    inum[lb[0]]+=1