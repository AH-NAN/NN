import os
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
def get_file(image_dir,label_dir):
    image_list=[]
    label_list=[1]
    for i in range(len(os.listdir(image_dir))-1):
        image_list.append(os.path.join(image_dir,'{:}.jpg'.format(i)))
    print(len(image_list))
    df=pd.read_csv(label_dir)
    label_list=label_list+list(df.values[:,1])
    print(len(label_list))
    return image_list,label_list
    

def onehot(labels):
    n_sample=len(labels)
    n_class=max(labels)+1
    onehot_labels=np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels]=1
    return onehot_labels

from vgg_preprocess import preprocess_for_train
#因为我们想对我们的图片进行与vgg16同样的预处理（否则网络不会有好效果）
#所以直接导入它的处理库，来进行处理
img_width=224
img_height=224
def get_batch(image_list,label_list,img_width,img_height,batch_size,capacity):
    #capacity内存中存储的最大容量
    image=tf.cast(image_list,tf.string)
    label=tf.cast(label_list,tf.int32)
    input_queue=tf.train.slice_input_producer([image,label])
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_contents,channels=3)
    image=preprocess_for_train(image,224,224)
    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
    label_batch=tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
