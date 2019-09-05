import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.livenessnet import LivenessNet
#LabelEncoder可以将标签分配一个0—n_classes-1之间的编码 
from sklearn.preprocessing import LabelEncoder
#scikit-learn中的一个函数，用于构建用于训练和测试的数据拆分
from sklearn.model_selection import train_test_split
#同样来自scikit-learn，该工具将生成关于模型性能的简要统计报告。加入混淆矩阵
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#用于执行数据扩充，为我们提供批量随机突变的图像。
from keras.preprocessing.image import ImageDataGenerator
#一个适用于此模型的优化器。（替代方案包括SGD，RMSprop等）。
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc
#从我的imutils包中，这个模块将帮助我们收集磁盘上所有图像文件的路径。
from imutils import paths
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-td", "--testset", required=True,
    help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
    help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
    help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="ROC.png",
    help="path to output ROC plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 8
EPOCHS = 50
print("[INFO] loading images...")
testimagePaths = list(paths.list_images(args["testset"]))
testdatas = []
testlabels = []
for imagePath in testimagePaths:
    # extract the class label from the filename, load the image and
    # resize it to be a fixed 32x32 pixels, ignoring aspect ratio
    testlabel = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    if isinstance(image, np.ndarray):
        pass
    else:
        continue
    image = cv2.resize(image, (32, 32))

    # update the data and labels lists, respectively
    # data ->>(311, 32, 32, 3)
    testdatas.append(image)
    # lables >> 311
    testlabels.append(testlabel)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
testdatas = np.array(testdatas, dtype="float") / 255.0
# testdatas = np.array(testdatas, dtype="float")
# encode the testlabels (which are currently strings) as integers and then
# one-hot encode them
testX = testdatas

le = pickle.loads(open(args["le"], "rb").read())
testlabels = le.fit_transform(testlabels)
testlabels = np_utils.to_categorical(testlabels, 2)
testY = testlabels

model = load_model(args["model"])
#,argmax 返回最大值所的索引，0：按列计算，1：行计算
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
# print(predictions)
# print(predictions[:,1:])
predictChose = []
for pre in predictions[:,1:]:
    if pre > 0.5:
        predictChose.append(1)
    else:
        predictChose.append(0)
print('*'*20)
# print(predictions.argmax(axis=1))
print(classification_report(testY.argmax(axis=1),
    predictChose, target_names=le.classes_))
tn, fp, fn, tp = confusion_matrix(testY.argmax(axis=1), predictChose).ravel()
print("验证验证")
RECALL = tp / (tp + fn)
FPR = fp / (fp + tn)
# BER --> balanced error rate
BER = 0.5 * (FPR + (1 - RECALL))
print("Balanced error rate:", BER)
#########################################################
fpr, tpr, thresholds = roc_curve(testY.argmax(axis=1),predictions[:,1:])
roc_auc = auc(fpr, tpr)
##确定最佳阈值
for i in range(len(fpr)):
    if fpr[i] + tpr[i] >= 1:
        i = i -1
        break
print("The best threshold:", fpr[i])

## 绘制roc曲线图
# 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f) optimum threshold = %f' % ( roc_auc, thresholds[i]))
# 画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE IN NUAA')
plt.legend(loc="lower right")
plt.savefig(args["plot"])
