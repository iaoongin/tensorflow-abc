from __future__ import absolute_import, division, print_function, unicode_literals

# 导入TensorFlow和tf.keras
import tensorflow as tf
from tensorflow import keras

from basic_classification import FashionMnistData

# ### 导入辅助库
# Python的一种开源的数值计算扩展
import numpy as np
#  Python 的 2D绘图库
# pyplot 是 Matplotlib 软件包中子包，提供了一个类似MATLAB的绘图框架。
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

# ### 导入Fashion MNIST 的离线数据集

(train_images, train_labels), (test_images, test_labels) = FashionMnistData.load_data()

# 加载数据集并返回四个NumPy数组:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("=========== train ===========")
# 训练集中有60,000个图像，每个图像表示为28 x 28像素
print(train_images.shape)

# 训练集中有60,000个标签:
print(len(train_images))

# 训练集每个标签都是0到9之间的整数:
print(train_labels)

print("=========== test ===========")
# 训练集中有60,000个图像，每个图像表示为28 x 28像素
print(test_images.shape)

# 训练集中有60,000个标签:
print(len(test_images))

# 训练集每个标签都是0到9之间的整数:
print(test_labels)

# ### 数据预处理

# 在训练网络之前必须对数据进行预处理。
# 如果您检查训练集中的第一个图像，您将看到像素值落在0到255的范围内:
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 在馈送到神经网络模型之前，我们将这些值缩放到0到1的范围。
# 为此，我们将像素值值除以255。
# 重要的是，对训练集和测试集要以相同的方式进行预处理:


train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示训练集中的前25个图像，并在每个图像下方显示类名。
# 验证数据格式是否正确，我们是否已准备好构建和训练网络。

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# ### 构建模型
#
# 构建神经网络需要配置模型的层，然后编译模型。
#
# 设置网络层
# 一个神经网络最基本的组成部分便是网络层。网络层从提供给他们的数据中提取表示，并期望这些表示对当前的问题更加有意义
# 大多数深度学习是由串连在一起的网络层所组成。大多数网络层，例如tf.keras.layers.Dense，具有在训练期间学习的参数。
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# ### 编译模型
# 在模型准备好进行训练之前，它还需要一些配置。这些是在模型的编译(compile)步骤中添加的:
#
# 损失函数 —这可以衡量模型在培训过程中的准确程度。 我们希望将此函数最小化以"驱使"模型朝正确的方向拟合。
# 优化器 —这就是模型根据它看到的数据及其损失函数进行更新的方式。
# 评价方式 —用于监控训练和测试步骤。以下示例使用准确率(accuracy)，即正确分类的图像的分数。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ### 训练模型
model.fit(train_images, train_labels, epochs=5)
# 随着模型训练，将显示损失和准确率等指标。该模型在训练数据上达到约0.88(或88％)的准确度。
#
# 评估准确率
# 接下来，比较模型在测试数据集上的执行情况:
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# ### 进行预测
#
# 通过训练模型，我们可以使用它来预测某些图像。
predictions = model.predict(test_images)

print(predictions[0])

# 预测是10个数字的数组。这些描述了模型的"信心"，即图像对应于10种不同服装中的每一种。我们可以看到哪个标签具有最高的置信度值：

print(np.argmax(predictions[0]))

print(test_labels[0])


# 用图表来查看全部10个类别


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 让我们看看第0个图像，预测和预测数组。
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# 让我们绘制几个图像及其预测结果。
# 正确的预测标签是蓝色的，不正确的预测标签是红色的。
# 该数字给出了预测标签的百分比(满分100)。请注意，即使非常自信，也可能出错。
#
# 绘制前X个测试图像，预测标签和真实标签
# 以蓝色显示正确的预测，红色显示不正确的预测
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# 最后，使用训练的模型对单个图像进行预测。
# 从测试数据集中获取图像
img = test_images[0]

print(img.shape)

# 将图像添加到批次中，即使它是唯一的成员。
img = (np.expand_dims(img,0))

print(img.shape)

# 现在来预测图像:

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)

print(class_names[prediction_result])





