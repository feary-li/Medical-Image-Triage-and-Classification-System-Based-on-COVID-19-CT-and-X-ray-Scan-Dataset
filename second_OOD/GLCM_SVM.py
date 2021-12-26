import skimage.feature
import skimage.segmentation
import skimage.data
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import warnings
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def feature(files, x, bins, y):
    for file in files:
        names = read(file)
        for file0 in names:
            image = io.imread(file + '/' + file0)
            if len(image.shape) == 3:
                image = image[:, :, :1]
                image = np.array(image).reshape(image.shape[0], image.shape[1])
            else:
                image = image
            img = skimage.feature.graycomatrix(image, [5], [0], 256, symmetric=True, normed=True)
            hist = np.array([skimage.feature.graycoprops(img, 'contrast'), skimage.feature.graycoprops(img, 'dissimilarity'),
                            skimage.feature.graycoprops(img, 'homogeneity'), skimage.feature.graycoprops(img, 'energy'),
                            skimage.feature.graycoprops(img, 'correlation'), skimage.feature.graycoprops(img, 'ASM')]).reshape(1, 6)
            # hist, bins = np.histogram(img, bins=bins, range=(0, 10250))
            x = np.vstack((x, hist))

    for i in range(0, len(files)):
        names = read(files[i])
        label = i*np.ones([len(names), 1])
        y = np.append(y, label, axis=0)

    return x, y


def read(path):
    filenames = os.listdir(path)
    return filenames


def grid(data_x, data_y):
    param_grid = {'C': [0.1, 0.5, 1, 2, 4], 'gamma': [0.02, 0.04, 0.06, 0.08, 0.1]}

    classifier = SVC('rbf')
    grid_search = GridSearchCV(classifier, param_grid)

    grid_search.fit(data_x, data_y)
    print('The parameters of the best model are: ')
    print(grid_search.best_params_)


def detect():
    Files = ['dataset/train/CT1', 'dataset/train/CT2', 'dataset/train/CT4']
    Bins = 200
    Data, y_label = np.zeros(6), np.zeros([1, 1])
    x_train, y_train = feature(Files, Data, bins=Bins, y=y_label)

    x_train, y_train = np.delete(x_train, [0], axis=0), np.delete(y_train, [0], axis=0)

    print('successfully read train dataset')

    test_files = ['dataset/test/CT1', 'dataset/test/CT2', 'dataset/test/CT4']
    x_test, y_test = feature(test_files, Data, bins=Bins, y=y_label)

    x_test, y_test = np.delete(x_test, [0], axis=0), np.delete(y_test, [0], axis=0)

    print('successfully read test dataset')
    # standard = StandardScaler()
    # x_train = standard.fit_transform(x_train)
    # x_test = standard.fit_transform(x_test)
    #
    # print('successfully standard dataset')

    # grid(x_test, y_test)

    clf = SVC(kernel='poly', C=10, gamma=0.02, degree=2)
    clf.fit(x_train, y_train)

    score = clf.score(x_test, y_test)
    score2 = clf.score(x_train, y_train)

    print(score)
    print(score2)
    print(clf.predict(x_test))

    print(classification_report(y_test, clf.predict(x_test)))

    sns.set()
    f, ax = plt.subplots()

    C2 = confusion_matrix(y_test, clf.predict(x_test), labels=[0, 1])

    sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴

    plt.show()
