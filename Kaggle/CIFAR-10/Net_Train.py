
# coding: utf-8

# In[ ]:


from mxnet import gluon
from mxnet import image
from Read_Data import ReadData
from Read_Data import Image_Show
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

Path_Label = '../../../Data/'
File_Name = 'Train_CIFAR10.csv'


# In[ ]:


# LoadImageData = ReadData(File_Name, Path_Label)
# print(LoadImageData.shape)


# ### 此处根据batch_size进行数据的分组（包括测试与batch）
#  此处根据实际需要（AlexNet的最小输入）需要对Image进行resize

# In[ ]:


def loadTrainTestData(Data, batch_size, Train_ratio = 0.8,origin_size=32, resize = None):
    # 测试数据与训练数据的数量
    Num_Train = int(len(Data) * Train_ratio)
    Num_Test = len(Data) - Num_Train
    print(Num_Train, Num_Test)
    # 检测是否可被batch_size整除
    if Num_Train%batch_size: # | Num_Test%batch_size:
        print("batch_size is not suitable!")
        return -1
    Batch_Train = int(Num_Train/batch_size)
    # 此处进行图像的resize
    Image = []
    if resize:
        for i in range(len(Data)):
            # print(Data[i,1:].shape)
            # print(Data[i,1:].reshape(3, 32, 32).transpose(2,1,0).shape)
            # print(misc.imresize(Data[i,1:].reshape(3, 32, 32), (resize,resize)))
            Image.append(misc.imresize(Data[i,1:].reshape(3, origin_size, origin_size), (resize,resize)).flatten())
            # Image.append(np.hstack((Data[i,:1], misc.imresize(Data[i,1:].reshape(3, 32, 32), (resize,resize)).flatten())))
        Image = np.array(Image)
        Image_Data = np.concatenate((Data[:,:1], Image), axis = 1).astype(np.uint8)
        # 之前出现颜色问题是因为整型和浮点型的原因
        plt.imshow(Image_Data[1,1:].reshape(resize, resize, 3))
        plt.show()
        Data = Image_Data;
    # Batch_Test = int(Num_Test/batch_size)
    # 将训练数据放入Batch_Train个batch中
    Train_Data = []
    for i in range(Batch_Train):
        Train_Data.append(Data[i*batch_size:(i+1)*batch_size, :])
    # 将剩下的数据作为Test_Data
    Test_Data = Data[Batch_Train* batch_size: , :]
    return np.array(Train_Data), np.array(Test_Data)


# In[ ]:


# Data = np.ones((100, 3073))
# print(Data.shape)
# Train_Data, Test_Data = loadTrainTestData(LoadImageData, 256, resize=224)


# In[ ]:


#plt.imshow(Image[2])
#plt.show()
# print(LoadImageData.shape)
# print(Train_Data.shape)
# print(Test_Data.shape)


# In[ ]:


def GetImageAndLabel(Data, size = 32):
    Label = Data[:,0]
    Image = Data[:,1:].reshape((len(Data),size, size, 3)).transpose((0,3,2,1))
    #print(Image.shape)
    return Image/255, Label


# In[ ]:


# Image, Label = GetImageAndLabel(Train_Data[0])
# print(Image.shape)
# print(Label.shape)


# In[ ]:


def Image_show(Image, Label):
    plt.imshow(Image.transpose(1,2,0))
    plt.title(Label)
    plt.show()


# In[ ]:


# print(Image[0].shape)
# Image_show(Image[0], Label[0])
#image_resize = misc.imresize(Image[0], (224,224))
#print(image_resize.shape)
#Image_show(image_resize.transpose(2,1,0), Label[0])


# ### 定义一个基础的神经网络

# In[ ]:


from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet.gluon.model_zoo import vision
from mxnet import autograd as ag
from time import clock
import mxnet
# ctx = mxnet.gpu()
# # AlexNet对输入的图形的像素尺寸有要求，不能用原先的32*32
# # AlexNet最后的全连接层需要根据实际需要，进行类别选择
# AlexNet = vision.alexnet(classes=10)
# #print(AlexNet)
# AlexNet.initialize(init=init.Xavier(),ctx=ctx)

# # loss = gluon.loss.SoftmaxCrossEntropyLoss()
# # Trainer = gluon.Trainer(ConvNet.collect_params(), 'sgd', {'learning_rate': 0.05})

# # x = nd.random_normal(shape=(10, 3, 224, 224))
# # print(x.shape)
# # y = AlexNet(x)
# # print(y.shape)

# from mxnet import gluon
# from mxnet import init
# from mxnet.gluon import nn
# ConvNet = nn.Sequential()
# with ConvNet.name_scope():
#     ConvNet.add(
#         nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         nn.Flatten(),
#         nn.Dense(128, activation="relu"),
#         nn.Dense(10)
#     )

# ConvNet.initialize(init=init.Xavier(), ctx=ctx)



# loss = gluon.loss.SoftmaxCrossEntropyLoss()
# Trainer = gluon.Trainer(AlexNet.collect_params(), 'sgd', {'learning_rate': 0.05})


# In[ ]:


def accuracy(out, label):
    return nd.mean(out.argmax(axis=1) == label).asscalar()


# In[ ]:


def evaluate_accuracy(Data, net, ctx, Size):
    test_image, test_label = GetImageAndLabel(Data, size= Size)
    test_image = nd.array(test_image, ctx=ctx)
    test_label = nd.array(test_label, ctx=ctx)
    out = net(test_image)
    # print(out.argmax(axis=1), " ", test_label)
    acc = accuracy(out, test_label)
    #test_image.asnumpy()
    #test_label.asnumpy()
    return acc


# In[ ]:


def train(train_data, test_data, net, echoes, loss_func, trainer, size, ctx, Test_Flag =False, lr = None):
    test_acc = 0
    for echoe in range(echoes):
        train_loss = 0
        train_acc = 0
        start_time = clock()
        if lr:
                lr *= 0.98
                trainer.set_learning_rate(lr * 0.98)
        for train_batch in train_data:
            #print(train_batch.shape)
            batch_size = len(train_batch)
            train_image, train_label = GetImageAndLabel(train_batch, size=size)
            #print(train_image.shape)
            #print(train_label.shape)
            with ag.record():
                train_image = nd.array(train_image, ctx=ctx)
                train_label = nd.array(train_label, ctx=ctx)
                #print(train_image[0])
                out = net(train_image)
                #print(out.shape)
                #print(out.argmax(axis=1))
                #print(train_label)
                loss = loss_func(out, train_label)
                #print(loss)
            # 此处需要backward()！！！！
            loss.backward()
            trainer.step(batch_size)
            mxnet.nd.waitall()
            train_loss += np.mean(loss.asnumpy())
            # 局部正确率的计算
            #print(train_label.shape)
            #print(batch_size)
            #print(out.argmax(axis=1))
            #print(train_label)
            #print(accuracy(out, train_label)/batch_size)
            train_acc += accuracy(out, train_label)
            # print(train_acc)
            train_image.asnumpy()
            train_label.asnumpy()
        # 此处需要输出最后的loss及其精度
        end_time = clock()
        use_time = (end_time - start_time) #/1000000
        if Test_Flag:
            test_acc = evaluate_accuracy(test_data, net, ctx, size)
        print("Echoe is %d, Train Loss is %f, Train Acc is %f, Test Acc is %f, Use-time is %f S, lr is %f" % (echoe+1, train_loss/len(train_data), train_acc/len(train_data), test_acc, use_time, lr))
        
        


# In[ ]:


# train(Train_Data, Test_Data, AlexNet, 50, loss, Trainer, 224, ctx=ctx)


# In[ ]:


# temp_out = nd.array([[-0.29596519,0.78109169,0.3583461,0.3933138,0.10683483,-0.18825993,-0.17625771,-0.33768478,-0.68717098,0.20490324],
#                     [-0.29596519,0.78109169,0.8583461,0.3933138,0.10683483,-0.18825993,-0.17625771,-0.33768478,-0.68717098,0.20490324],
#                     [-0.29596519,0.78109169,0.3583461,0.9933138,0.10683483,-0.18825993,-0.17625771,-0.33768478,-0.68717098,0.20490324]])
# print(temp_out.shape)
# print(temp_out.argmax(axis=1))


# ## 调试网络

# In[ ]:


# def softmax(X):
#     exp = nd.exp(X)
#     # 假设exp是矩阵，这里对行进行求和，并要求保留axis 1，
#     # 就是返回 (nrows, 1) 形状的矩阵
#     partition = exp.sum(axis=1, keepdims=True)
#     return exp / partition
# def cross_entropy(yhat, y):
#     return - nd.pick(nd.log(yhat), y)


# In[ ]:


# import sys
# sys.path.append('../../../gluon-tutorials-zh')
# from mxnet import ndarray as nd
# from mxnet import autograd
# import utils

# batch_size = 256
# train_data, test_data = utils.load_data_fashion_mnist(batch_size)


# In[ ]:


# from mxnet import gluon
# from mxnet import init
# from mxnet.gluon import nn
# ConvNet = nn.Sequential()
# with ConvNet.name_scope():
#     ConvNet.add(
#         nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         nn.Flatten(),
#         nn.Dense(128, activation="relu"),
#         nn.Dense(10)
#     )
# ConvNet.initialize(init=init.Xavier())


# In[ ]:


# softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.005})


# In[ ]:


# from mxnet import ndarray as nd
# from mxnet import autograd

# lr = 0.5

# for epoch in range(100):
#     test_acc = 0.
#     train_loss = 0.
#     train_acc = 0.
#     #for data, label in train_data:
#     for train_batch in Train_Data:
#         data, label = GetImageAndLabel(train_batch, size=32)
#         data = nd.array(data)
#         #print(data.shape)
#         label = nd.array(label)
#         #print(label.shape)
#         batch_size = len(data)
#         with autograd.record():
#             #print(data.shape)
#             output = net(data)
#             #temp_soft = softmax(output)
#             #temp_loss = cross_entropy(temp_soft, label)
#             #print(output)
#             #print(output.argmax(axis=1))
#             #print(label)
#             #print(temp_soft)
#             #print(temp_loss)
            
#             #print(label.shape)
            
#             # loss计算有问题
#             loss = softmax_cross_entropy(output, label)
#             #print(loss)
#             #print(loss.shape)
#         loss.backward()
#         trainer.step(batch_size)
        
#         trainer.set_learning_rate(lr)
#         train_loss += nd.mean(loss).asscalar()
#         train_acc += accuracy(output, label)
#     lr = lr * 0.9
#     test_acc = evaluate_accuracy(nd.array(Test_Data), net)
#     print("Epoch %d. Loss: %f, Train acc %f, Test acc %f, lr is %f" % (
#         epoch, train_loss/len(Train_Data), train_acc/len(Train_Data), test_acc, lr))

