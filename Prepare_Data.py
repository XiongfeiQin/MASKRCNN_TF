'''
convert images and labels to tfrecords
labels are of shape [20,20,3]
images are of [160,160]
'''

# -*- coding = utf-8 -*-
import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
import numpy as np

# make tfrecord file
# cwd = '/home/higo/DataSet/1030/'
# cwd = 'E:/fingerprint/1030/'
# cwd = 'E:/fingerprint/1216/'
# cwd = 'E:/fingerprint/1222/'
cwd = 'E:/fingerprint/0320/'
classes = ('labels1', 'pic') #人为 设定 2 类

label = classes[0]
name = classes[1]
# enumerate() format: 0 name1 \n 1 name2 ......
# index & name show in pairs...
class_path = cwd+name+'/'  # images
label_path = cwd+label+'/'  # labels

writer = tf.python_io.TFRecordWriter(cwd+"train_data3.tfrecords") #要生成的文件
for i in range(96001):  # 生成训练集
    img_name = str(i)+'.bmp'
    label_name = str(i)+'.txt'
    dirc_path = label_path + label_name
    # dirc = Image.open(dirc_path)
    # dirc = dirc.resize(1, 800)
    dirc = np.loadtxt(dirc_path)
    dirc = np.reshape(dirc, [20, 20, 3])
    dirc_raw = dirc.tobytes()

    img_path = class_path+img_name #每一个图片的地址
    img = Image.open(img_path)
    img = img.resize((160, 160))
    # ImgMean = ImageStat.Stat(img).mean
    # ImgStd = ImageStat.Stat(img).stddev
    # print(ImgMean, ImgStd)
    # img = (img-ImgMean)/ImgStd
    # print(img)
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()#将图片转化为二进制格式

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dirc_raw])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    })) #example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  #序列化为字符串
    print("this is num %d" % i)
writer.close()

writer1 = tf.python_io.TFRecordWriter(cwd+"test_data3.tfrecords") #要生成的文件
for i in range(96001, len(os.listdir(label_path))):  # 生成测试集
    img_name = str(i)+'.bmp'
    label_name = str(i)+'.txt'
    dirc_path = label_path + label_name
    # dirc = Image.open(dirc_path)
    # dirc = dirc.resize(1, 800)
    dirc = np.loadtxt(dirc_path)
    dirc = np.reshape(dirc, [20, 20, 3])
    dirc_raw = dirc.tobytes()
    img_path = class_path+img_name #每一个图片的地址
    img = Image.open(img_path)
    img = img.resize((160, 160))
    # resize input: The requested size in pixels, as a 2-tuple: (width, height).
    # won't change the dimension of the picture...
    img_raw = img.tobytes()#将图片转化为二进制格式

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dirc_raw])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    })) #example对象对label和image数据进行封装
    writer1.write(example.SerializeToString())  #序列化为字符串
    print("this is num %d" % i)
writer1.close()
