# USAGE
# python train_liveness.py --dataset dataset --model liveness.model --le le.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
# from pyimagesearch.livenessnet import LivenessNet
from pyimagesearch.livenessnet import LivenessYXMNet
# from pyimagesearch.livenessnet import LivenessVGG16
from sklearn.preprocessing import LabelEncoder   # LabelEncoder可以将标签分配一个0—n_classes-1之间的编码 
from sklearn.model_selection import train_test_split  # scikit-learn中的一个函数，用于构建用于训练和测试的数据拆分
from sklearn.metrics import classification_report  # 同样来自scikit-learn，该工具将生成关于模型性能的简要统计报告
from keras.preprocessing.image import ImageDataGenerator # 用于执行数据扩充，为我们提供批量随机突变的图像
from keras.optimizers import Adam  # 一个适用于此模型的优化器（替代方案包括SGD，RMSprop等）
from keras.utils import np_utils  
from imutils import paths  # 从imutils包中，这个模块将帮助我们收集磁盘上所有图像文件的路径
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-td", "--trainset", required=True,
	help="path to input dataset")
ap.add_argument("-ed", "--testset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of
# epochs to train for
# need to fine-tune
INIT_LR = 1e-4
BS = 4
EPOCHS = 50

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
trainImagePaths = list(paths.list_images(args["trainset"]))
testImagePaths = list(paths.list_images(args["testset"]))
train_data = []
test_data = []
train_labels = []
test_labels = []

for imagePath in trainImagePaths:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
	train_label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)

	if isinstance(image, np.ndarray):
		pass
	else:
		continue

	image = cv2.resize(image, (32, 32))

	# update the data and labels lists, respectively
	train_data.append(image)
	train_labels.append(train_label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]

# normalization
train_data = np.array(train_data, dtype="float") / 255.0
# no normalization
# train_data = np.array(train_data, dtype="float") 
# print(train_data.shape)

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
train_labels = np_utils.to_categorical(train_labels, 2)
# print(train_labels.shape)

# Do the same thing with test dataset
for imagePath in testImagePaths:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
	test_label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)

	if isinstance(image, np.ndarray):
		pass
	else:
		continue

	image = cv2.resize(image, (32, 32))

	# update the data and labels lists, respectively
	test_data.append(image)
	test_labels.append(test_label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]

# normalization
test_data = np.array(test_data, dtype="float") / 255.0
# no normalization
# test_data = np.array(test_data, dtype="float")
# print(train_data.shape)

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
test_labels = le.fit_transform(test_labels)
test_labels = np_utils.to_categorical(test_labels, 2)
# print(test_labels.shape)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, validX, trainY, validY) = train_test_split(train_data, train_labels,
	test_size=0.25, random_state=42)
testX = test_data
testY = test_labels


# construct the training image generator for data augmentation

# rotation_range 角度值，0~180，图像旋转
# zoom_range 随机缩放范围
# width_shift_range 水平平移，相对总宽度的比例
# height_shift_range 垂直平移，相对总高度的比例
# shear_range 随机错切换角度
# horizontal_flip 一半图像水平翻转
# fill_mode="nearest" 填充新创建像素的方法
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# No augumentation
# aug = ImageDataGenerator()

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessYXMNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(validX, validY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS) # 以batch_size来预测，但是是全部预测完，即predictions的shape和testY的shape是一样的
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# save the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])