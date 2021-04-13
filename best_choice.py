import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import os
import numpy as np


'''
利用图像相似度筛选出T1,T2信号照片并保存
原图像为初步分类后的图像，test/1,test/2
保存到test/T1,test/T2
模板图像datas/T1_s,datas/T2_s
模板可换(*>*)
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


# 计算图片相似度ORB算法
def orb_compare(img_1,img_2):
    # 初始化orb检测器
    orb = cv.ORB_create()
    kp1,des1=orb.detectAndCompute(img_1,None)
    kp2,des2=orb.detectAndCompute(img_2,None)

    # 提取并计算特征点
    bf=cv.BFMatcher(cv.NORM_HAMMING)
    # 利用knn来筛选结果
    matches=bf.knnMatch(des1,trainDescriptors=des2,k=2)
    # 查看最大匹配点数目，m距离小于0.9*n距离，可以修改0.9
    good = [m for (m, n) in matches if m.distance < 0.9*n.distance]
    # 计算相似率，越大相似程度越高
    sim_1=len(good)/len(matches)
    return sim_1

# 利用opencv中的structural_similarity函数进行相似度计算
def ssim_compare(img_1,img_2):
    score = ssim(img_1, img_2,multichannel=True)
    return score

# 计算差值哈希算法
def dHash(img):
    # 将图片大小调整为9*8
    img=cv.resize(img,(9,8))
    hash_str0=[]
    # 每行前一个像素大于后一个像素为1，相反为0
    for i in range(8):
        hash_str0.append(img[:,i]>img[:,i+1])
    hash_str1=np.array(hash_str0)+0
    hash_str2=hash_str1.T
    hash_str3=hash_str2.reshape(1,-1)[0].tolist()
    dhash_str=''.join([str(x) for x in hash_str3])
    return dhash_str


# 计算汉明距离
def hammingDist(s1,s2):
    assert len(s1)==len(s2)
    return sum([ch1 !=ch2 for ch1,ch2 in zip(s1,s2)])


# 选择最佳匹配函数，传入模板路径，测试文件夹路径，保存路径
# 利用上面三种方法进行选择
def best_choice(model_path,test_path,save_path):
    # 加载模板图片和测试图片
    img_s=load_images(model_path)[0]
    tests=load_images(test_path)
    print(len(tests))
    # 建立4个列表保存匹配算法的返回的相似率，orb算法，ssim算法，dhash算法，综合三种算法
    sim_1s=[]
    sim_2s=[]
    hash_scores = []
    sim_sum = []
    # 遍历测试集中所有图片进行比较
    for i in range(len(tests)):
        sim_1 = orb_compare(img_s, tests[i])
        sim_2 = ssim_compare(img_s, tests[i])
        dhash_str1 = dHash(img_s)
        dhash_str2 = dHash(tests[i])
        p_score = 1 - hammingDist(dhash_str1, dhash_str2) * 1 / (32 * 32 / 4)

        sim_1s.append(sim_1)
        sim_2s.append(sim_2)
        hash_scores.append(p_score)
        sim_sum.append(0.7*sim_1+0.1* sim_2+0.2*p_score)

    # 查找四种算法得到的最大相似率的索引
    sim1max_index=sim_1s.index(max(sim_1s))
    sim2max_index=sim_2s.index(max(sim_2s))
    max_hamming_index = hash_scores.index(max(hash_scores))
    maxsum_index = sim_sum.index(max(sim_sum))

    # print(maxsum_index)
    # 根据最大相似率的索引保存匹配到的图片
    cv.imwrite(save_path + "1.jpg", tests[sim1max_index])
    cv.imwrite(save_path + "2.jpg", tests[sim2max_index])
    cv.imwrite(save_path + "3.jpg", tests[max_hamming_index])
    cv.imwrite(save_path + "4.jpg", tests[maxsum_index])

    print("save sucess")

# 匹配T1信号保存到test/T1中
# 匹配T2信号保存到test/T2中
best_choice("datas/T1_s/","test/1/","test/T1/")
best_choice("datas/T2_s/","test/2/","test/T2/")

print("SUCESSFUL")



cv.waitKey()
cv.destroyAllWindows()







