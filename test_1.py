from sklearn.externals import joblib
import cv2 as cv
import numpy as np
from skimage import feature as skft

'''
加载模型测试一张照片的类别
照片位置datas/test_images
'''



def load_img(img_path):
    img = cv.imread(img_path,0)
    img = cv.resize(img,(512,512),interpolation=cv.INTER_LINEAR)
    return img

def lbp(img):
    hist = np.zeros((1, 12))
    lbp=skft.local_binary_pattern(img,P=10,R=4,method='uniform')
    max_bins=int(lbp.max()+1)
    hist[0], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    # hist=np.array(hist).reshape(-1,1)
    return hist

img=load_img('datas/test_images/7.jpg')
cv.imshow('img',img)
test=lbp(img)

# 加载所保存的模型
clf=joblib.load('clf.pkl')
y=clf.predict(test)
y=int(y)
print(y)
if y==0:
    print("不是所选样本")
elif y==1:
    print("T1信号样本")
else:
    print("T2信号样本")
cv.waitKey()
cv.destroyAllWindows()