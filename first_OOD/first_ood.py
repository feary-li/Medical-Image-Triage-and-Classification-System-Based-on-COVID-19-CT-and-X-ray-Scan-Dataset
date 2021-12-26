import os
from PIL import Image
import numpy as np
import time

start = time.time()

# 读取数据

def get_X(data):
    inp = data
    return inp[:, :]

def loader():
    train_data = []
    y_train = []
    for cla in range(1,2):
        file_path = 'dataset/train/CT'
        img_dirs = os.listdir(file_path)
        for item in img_dirs:
            img_dir = os.path.join(file_path, item)
            img = Image.open(img_dir).getdata(band=0)
            img = img.resize((28, 28))
            img = np.array(img)
            img = img.reshape(28 * 28)
            img = list(img)
            img.append(cla)
            train_data.append(img)
            y_train.append(0)

    for cla in range(1,2):
        file_path = 'dataset/train/X-ray'
        img_dirs = os.listdir(file_path)
        for item in img_dirs:
            img_dir = os.path.join(file_path, item)
            img = Image.open(img_dir).getdata(band=0)
            img = img.resize((28, 28))
            img = np.array(img)
            img = img.reshape(28 * 28)
            img = list(img)
            img.append(cla)
            train_data.append(img)
            y_train.append(1)

    test_data = []
    y_test = []
    for cla in range(1,2):
        file_path = 'dataset/test'
        img_dirs = os.listdir(file_path)
        for item in img_dirs:
            img_dir = os.path.join(file_path, item)
            img = Image.open(img_dir).getdata(band=0)
            img = img.resize((28, 28))
            img = np.array(img)
            img = img.reshape(28 * 28)
            img = list(img)
            img.append(cla)
            test_data.append(img)
        for item in img_dirs:
            title = str(item)
            if 'X' in title:
                y_test.append(1)
            else:
                y_test.append(0)

    X_train = get_X(np.array(train_data))
    y_train = np.array(y_train).T

    X_test= get_X(np.array(test_data))
    y_test = np.array(y_test).T
    # print(y_test)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    return X_train, y_train, X_test, y_test

# 进行分类
def first_OOD():
    X_train, y_train, X_test, y_test = loader()
    
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    
    classifier = SVC(kernel='rbf', C=0.1)
    
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print("准确率为 %2.3f" % (accuracy * 100.0))
    print(classification_report(y_test,y_pred))
    
    # 进行图片的删除
    d_l = []
    for i in range(np.array(y_pred).shape[0]):
        if np.array(y_pred)[i] == 1:
            d_l.append(i)
    
    i = 0
    d_I = []
    file_path = 'dataset/test'
    
    for cla in range(np.array(d_l).shape[0]):
              img_dirs = os.listdir(file_path)
              for item in img_dirs:
                  # print(item)
                  img_dir = os.path.join(file_path, item)
                  i += 1
                  if i == d_l[cla]+1:
                    d_I.append(item)
                    break
              i = 0
    # print(d_I)
    for i in range(np.array(d_I).shape[0]):
       # os.remove(os.path.join(file_path, d_I[i]))
       print("Delete File: " + os.path.join(file_path, d_I[i]))
    
    print('calcurate time:{}'.format(time.time()-start))
