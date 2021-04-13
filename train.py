import cv2 as cv
import matplotlib.pyplot as plt
from skimage import feature as skft
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix,classification_report,precision_score
from sklearn.model_selection import train_test_split
from sklearn import svm


'''

使用LBP图像纹理特征+SVC进行图像初步分类
模型可重复训练
test_1.py文件对某一张照片进行检测
test_p.py文件对test中images中文件进行批量检测并分类保存在0，1，2文件夹中
1代表可能的T1信号图片
2代表可能的T2信号图片

'''

### 读取照片并将照片转换为灰度图片大小调整为512*512
def read_image(img_name):
    img=cv.imread(img_name,0)
    img = cv.resize(img, (512, 512), interpolation=cv.INTER_LINEAR)
    return img


### 从文件夹中加载照片，并返回图片列表
def load_images(path):
    images=[]
    for fn in os.listdir(path):                         # 遍历整个文件夹
        if fn.endswith('.jpg'):                         # 如果是fn是以.jpg结尾则得到完整路径
            fd = os.path.join(path, fn)
            images.append(read_image(fd))               # 读取对应文件并存储到images[]列表中
    return images


# 读取文件夹0，1，2中照片
images_x0=load_images('datas/train_datas/0')
images_x1=load_images('datas/train_datas/1')
images_x2=load_images('datas/train_datas/2')


# 转换为np数组类型
images_x0=np.array(images_x0)
images_x1=np.array(images_x1)
images_x2=np.array(images_x2)


# 根据照片数量，生成标签y0,y1,y2对应于0，1，2
y0=np.zeros((images_x0.shape[0],1))
y1=np.ones((images_x1.shape[0],1))
y2=np.full((images_x2.shape[0],1),2)

# 将训练照片连接到一个数组中，将训练标签连接到一个数组中
x=np.vstack((images_x0,images_x1,images_x2))
y=np.vstack((y0,y1,y2))

print(x.shape)
print(y.shape)


### 划分训练集和测试集数量
# 测试集数量为总数*test_size,random_state混洗数据
# 可以修改test_size,随机率random_state值来训练得到不同的准确率

# random_state=5   0.78
# random_state=4   0.86
# random_state=2   0.72
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.01,random_state=5)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# print(len(x_train))


### 获取图片LBP特征

def lbp_texture(train_data,test_data):

    # 生成训练，测试样本的直方图
    train_hist=np.zeros((len(x_train),12))
    test_hist=np.zeros((len(x_test),12))

    # 遍历训练样本的照片得到每张照片的LBP特征，归一化直方图并保存到train_hist中
    for i in np.arange(len(x_train)):

        # 得到LBP特征，p是依赖点个数，R为半径，method=['default','uniforn','nri_uniform','ror','var]
        # 可以修改P，R，method方法，得到不同准确率
        lbp=skft.local_binary_pattern(train_data[i],P=10,R=4,method='uniform')              #
        max_bins=int(lbp.max()+1)

        # 生成LBP特征归一化后直方图，作为SVC分类数据
        train_hist[i],_=np.histogram(lbp,density=True,bins=max_bins,range=(0,max_bins))


    for i in np.arange(len(x_test)):
        lbp = skft.local_binary_pattern(test_data[i], P=10, R=4, method='uniform')
        max_bins = int(lbp.max() + 1)
        test_hist[i], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))

    return train_hist,test_hist


# 得到训练数据，测试数据LBP特征直方图
x_train,x_test=lbp_texture(x_train,x_test)
print(x_train.shape)
print(x_test.shape)


# 设置SVC分类器，C为惩罚参数类型浮点数，kernel为核函数，用于非线性分类，gamma是rbf内核系数
# 可以设置不同的C，kernel函数，gamma值得到不同准确率，'ovo'用于多分类

clf=svm.SVC(C=6.3,kernel='rbf',gamma=20,decision_function_shape='ovo')
clf.fit(x_train,y_train)

# 保存训练模型clf.pkl
joblib.dump(clf,'./clf.pkl')
# 使用模型进行预测
p_test=clf.predict(x_test)


print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))
print('decision_function:\n', clf.decision_function(x_train))
print(precision_score(y_test, p_test, average='macro'))
print(recall_score(y_test, p_test, average='macro'))
print(f1_score(y_test, p_test, average='macro'))

# 训练准确率
print(accuracy_score(y_test, p_test))

