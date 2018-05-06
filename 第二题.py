方式一



import tensorflow as tf

import numpy as np
import gc
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
batch_size=1000
step为随机数，表示下标开始的位置，以此为基准选取1000行数据

        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
		
		
		
方式二
若数据是存放在问价夹里，每个文件夹里有若干文件的形式

import os
import random

rootdir = "d:\\face\\train"
file_names = []
for parent, dirnames, filenames in os.walk(rootdir):   
    file_names = filenames

x = random.randint(0, len(file_names)-1)
for x in range(x+1000):
    print(file_names[x])