import numpy as np
import torch
from numpy import linalg as LA
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import warnings
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import scipy.stats

warnings.filterwarnings("ignore")


class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.model = VGG16(weights='imagenet', pooling='max', include_top=False,
                           input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]))

    def extract_feat(self, img_path):
        '''提取图像特征

        :param img_path: 图像路径
        :return: 归一化后的图像特征
        '''
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat

    def set_name(self, path):
        name = os.listdir(path)
        return name

    def feature(self, dir):
        ct1 = self.set_name(dir)
        i = 0
        for images in ct1:
            if i == 0:
                feature_1 = self.extract_feat(dir + '/' + images)
            else:
                feature_1 = np.vstack((feature_1, self.extract_feat(dir + '/'+images)))
            i += 1
        return feature_1

    def guass(self, file):
        fea = self.feature(file)
        mu = np.mean(fea, axis=0)
        sigma = np.cov(fea)
        # sigma = 0
        # for i in range(len(fea[:, 0])):
        #     a = np.mat(fea[i, :]-mu)
        #     sigma = a.T * a + sigma
        # sigma = np.array(sigma) / len(fea[:, 0])
        return fea, mu, sigma

    def mahalanobis(self, dir_file):
        fea, mu, sigma = self.guass(dir_file)
        inv_sigma = np.linalg.inv(sigma)
        for i in range(len(fea[0, :])):
            if i == 0:
                d_1 = np.mat(fea[:, i] - mu[i])
                d_m = d_1 * np.mat(inv_sigma) * d_1.T
                d_m = np.array(d_m)
            else:
                d_1 = np.mat(fea[:, i] - mu[i])
                d_m = np.vstack((d_m, np.array(d_1 * np.mat(inv_sigma) * d_1.T)))
        return d_m, mu, sigma

    def cross_entropy(self, Y, P):
        Y = np.float_(Y)
        P = np.float_(P)
        return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

    def KL_divergence(self, p, q):
        return scipy.stats.entropy(p, q)

    def maha(self, fea_prd, mu, sigma):
        d_1 = fea_prd - mu
        inv_sigma = np.linalg.inv(sigma)
        inv_sigma = np.mat(inv_sigma)
        for i in range(len(d_1)):
            d = d_1[i] * np.ones(len(sigma[:, 0]))
            d = np.mat(d)
            if i == 0:
                mah = d * inv_sigma * d.T
                mah = np.array(mah)
            else:
                mah = np.vstack((mah, d * inv_sigma * d.T))

        return mah

    def predict(self, dir_prd, mu, sigma1, sigma2, target):
        name = self.set_name(dir_prd)
        prd = np.zeros(len(name))
        i = 0
        for images in name:
            feature_prd = self.extract_feat(dir_prd + '/' + images)
            CE = np.zeros(4)
            for j in range(len(CE)):
                if j == 0:
                    maha = self.maha(feature_prd, mu[j, :], sigma1)
                elif j == 1:
                    maha = self.maha(feature_prd, mu[j, :], sigma2)
                # elif j == 2:
                #     maha = self.maha(feature_prd, mu[j, :], sigma3)
                # else:
                #     maha = self.maha(feature_prd, mu[j, :], sigma4)
                maha = np.array(maha).reshape(512, 1)
                loss_output = self.KL_divergence(maha, target[:, j].reshape(512, 1))
                CE[j] = loss_output
            prd[i] = np.argmin(CE) + 1
            i += 1
        return prd

    def target(self, dir_tar, tar):
        name = self.set_name(dir_tar)
        return tar * np.ones(len(name))


def detect():
    model = VGGNet()
    CT1, mu1, sigma1 = model.mahalanobis('dataset/second_OOD/train/CT1')
    CT2, mu2, sigma2 = model.mahalanobis('dataset/second_OOD/train/CT2')
    # CT3, mu3, sigma3 = model.mahalanobis('dataset/second_OOD/train/CT3')
    # CT4, mu4, sigma4 = model.mahalanobis('dataset/second_OOD/train/CT4')

    mu = np.array([mu1, mu2])
    CT = np.hstack((CT1, CT2))

    test_p1 = 'dataset/second_OOD/test/CT1'
    test_p2 = 'dataset/second_OOD/test/CT2'
    # test_p3 = 'dataset/second_OOD/test/CT3'
    # test_p4 = 'dataset/second_OOD/test/CT4'

    target = np.hstack((model.target(test_p1, 1), model.target(test_p2, 2)))

    prd = np.hstack((model.predict(test_p1, mu, sigma1, sigma2, CT),
                     model.predict(test_p2, mu, sigma1, sigma2, CT)))

    # 热力图
    sns.set()
    f, ax = plt.subplots()

    C2 = confusion_matrix(target, prd, labels=[1, 2, 3, 4])

    sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴

    print(classification_report(target, prd))

    plt.show()

    
