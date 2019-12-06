import tensorflow as tf
import numpy as np

class vgg16:
    def __init__(self,image):
        self.parameters=[]#在类初始化时加入全局列表，将共享参数放入
        self.image=image
        self.convlayers()
        self.fc_layers()
        self.probs=tf.nn.softmax(self.fc8)#输出概率
    def saver(self):
        return tf.train.Saver()
    def maxpool(self,name,input_data):
        out=tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],
                          padding="SAME",name=name)
        return out
    def fc(self,name,input_data,out_channel,trainable=True):
        shape=input_data.get_shape().as_list()
        if len(shape)==4:
            size=shape[-1]*shape[-2]*shape[-3]#全连接层输入神经数
            #如果是图片batch，把他展开成一维
        else:
            size=shape[1]
        #全连接输入可能是卷积层的输出，对应那种图像的形式，[None,height,weight,channel]
        #也可能是另一个全连接层的输出，只有一维了
        input_data_flat=tf.reshape(input_data,[-1,size])#展开输入
        with tf.variable_scope(name):
            weights=tf.get_variable(name=name+"weights",
                                   shape=[size,out_channel],
                                    dtype=tf.float32,trainable=trainable)
            biases=tf.get_variable(name=name+"biases",shape=[out_channel],
                                   dtype=tf.float32,trainable=trainable)
            res=tf.matmul(input_data_flat,weights)
            out=tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters+=[weights,biases]#加入参数
        return out
    #我们选择微调fc，而卷积层不变
    def conv(self,name,input_data,out_channel,trainable=False):
        in_channel=input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel=tf.get_variable(name=name+"weights",
                                   shape=[3,3,in_channel,out_channel],
                                   dtype=tf.float32,trainable=trainable)
            biases=tf.get_variable(name=name+"biases",
                                   shape=[out_channel],
                                   dtype=tf.float32,trainable=trainable)
            conv_res=tf.nn.conv2d(input_data,kernel,[1,1,1,1],padding="SAME")
            res=tf.nn.bias_add(conv_res,biases)
            out=tf.nn.relu(res,name=name)
        self.parameters+=[kernel,biases]
        return out
    def convlayers(self):
        #conv1
        self.conv1_1=self.conv("conv1_1",self.image,64,trainable=False)
        self.conv1_2=self.conv("conv1_2",self.conv1_1,64,trainable=False)
        self.pool1=self.maxpool("pool1",self.conv1_2)
        #conv2
        self.conv2_1=self.conv("conv2_1",self.pool1,128,trainable=False)
        self.conv2_2=self.conv("conv2_2",self.conv2_1,128,trainable=False)
        self.pool2=self.maxpool("pool2",self.conv2_2)
        #conv3
        self.conv3_1=self.conv("conv3_1",self.pool2,256,trainable=False)
        self.conv3_2=self.conv("conv3_2",self.conv3_1,256,trainable=False)
        self.conv3_3=self.conv("conv3_3",self.conv3_2,256,trainable=False)
        self.pool3=self.maxpool("pool3",self.conv3_3)
        #conv4
        self.conv4_1=self.conv("conv4_1",self.pool3,512,trainable=False)
        self.conv4_2=self.conv("conv4_2",self.conv4_1,512,trainable=False)
        self.conv4_3=self.conv("conv4_3",self.conv4_2,512,trainable=False)
        self.pool4=self.maxpool("pool4",self.conv4_3)
        #conv5
        self.conv5_1=self.conv("conv5_1",self.pool4,512,trainable=False)
        self.conv5_2=self.conv("conv5_2",self.conv5_1,512,trainable=False)
        self.conv5_3=self.conv("conv5_3",self.conv5_2,512,trainable=False)
        self.pool5=self.maxpool("pool5",self.conv5_3)
    def fc_layers(self):
        self.fc6=self.fc("fc6",self.pool5,4096,trainable=False)
        self.fc7=self.fc("fc7",self.fc6,4096,trainable=False)
        #前面都是false，表示被冻结了，不训练。
        self.fc8=self.fc("fc8",self.fc7,2,trainable=True)#2代表要输出的类别数
        #仅仅微调最后一层的参数（微调：finetuining）
    #载入参数
    #weight_file='./vgg16/vgg16_weights.npz'
    #模型文件https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
    #npz是以键值对形式保存
    #分类文件https://www.cs.toronto.edu/~frossard/vgg16/#imagenet_classes.py
    def load_weights(self,weight_file,sess):
        weights=np.load(weight_file)
        keys=sorted(weights.keys())
        for i,k in enumerate(keys):
            if i not in [30,31]:#剔除不需要的层,对应fc8
                sess.run(self.parameters[i].assign(weights[k]))
        print("--------weights loaded-----------")