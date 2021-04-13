from sklearn.externals import joblib
import cv2 as cv
import numpy as np
from skimage import feature as skft
import os


'''
初步筛选
加载模型对照片进行批量分类，并分类保存
原照片位置test/images
分类保存位置test/0,test/1,test/2
'''


def read_image(img_name):
    img=cv.imread(img_name,0)
    img = cv.resize(img, (512, 512), interpolation=cv.INTER_LINEAR)
    return img

def load_images(path):
    images=[]
    for fn in os.listdir(path):
        if fn.endswith('.jpg'):
            fd = os.path.join(path, fn)
            images.append(read_image(fd))
    return images

test_images = load_images('test/images/')
test_images=np.array(test_images)

cv.imshow('img',test_images[0])

print(test_images.shape)

def lbp_texture(train_data):
    hist=np.zeros((len(test_images),12))

    for i in np.arange(len(test_images)):
        # 应与训练时参数相一致
        lbp=skft.local_binary_pattern(train_data[i],P=10,R=4,method='uniform')
        max_bins=int(lbp.max()+1)
        hist[i], _ = np.histogram(lbp,density=True,bins=max_bins,range=(0,max_bins))

    return hist

test_datas=lbp_texture(test_images)
clf=joblib.load('clf.pkl')

# 加载模型进行预测，得到y标签列表
y=clf.predict(test_datas)
y=y.tolist()
print(len(y))
print(y[0])

# 遍历y中标签，并对test照片分类保存
for i in range(len(y)):
    if y[i]==0.0:
        cv.imwrite("test/0/"+str(i)+".jpg",test_images[i])
    elif y[i]==1.0:
        cv.imwrite("test/1/"+str(i)+".jpg",test_images[i])
    else:
        cv.imwrite("test/2/" + str(i) + ".jpg", test_images[i])


print("sucessful")













