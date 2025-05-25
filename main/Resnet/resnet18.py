import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.layers import Activation,Add,Input,BatchNormalization,GlobalAveragePooling2D
from sklearn.model_selection import train_test_split


# 读取数据
def read_file(File='datasets2022.npz'):
    dataset = np.load(File)
    train_data = dataset['train'] #training data 30412
    test_data = dataset['test'] #testing data 7603
    label_train = dataset['label_train'] #label for training data 30412
    return train_data, test_data, label_train


# 封装Conv2D+BatchNormalization+Relu，每个卷积操作后加一个BN层
def Conv_BN_Relu(filters, kernel_size, strides, input_layer):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


# 实线(a)和虚线(b)两种residual结构
def res_a_or_b(input_x, filters, flag):
    # residual-a不对支路进行处理
    if flag == 'a':
        # 主路
        x = Conv_BN_Relu(filters, (3,3), 1, input_x)
        x = Conv_BN_Relu(filters, (3,3), 1, x)
        # 输出
        y = Add()([x, input_x])
        return y
    # residual-b将支路尺寸减半，以使支路输出与主路输出矩阵shape相同
    elif flag == 'b':
        # 主路
        x = Conv_BN_Relu(filters, (3,3), 2, input_x)
        x = Conv_BN_Relu(filters, (3,3), 1, x)
        # 支路
        input_x = Conv_BN_Relu(filters, (1,1), 2, input_x)
        # 输出
        y = Add()([x, input_x])
        return y
        

train_data, test_data, label_train = read_file()
# 对训练数据水平、垂直翻转
temp1 = np.zeros_like(train_data)
temp2 = np.zeros_like(train_data)
for i in range(len(train_data)):
    temp1[i] = tf.image.flip_left_right(train_data[i])
    temp2[i] = tf.image.flip_up_down(train_data[i])
train_data = np.concatenate((train_data, temp1, temp2))
label_train = np.concatenate((label_train, label_train, label_train))


# 调试时 用以划分训练集和验证集
x_train,x_test,y_train,y_test = train_test_split(train_data,label_train,
                                                 test_size=0.2,random_state=5)
    

img_x, img_y = 52, 52

#print(y_train.shape)


#---------ResNet--------------#
# conv1
input_layer = Input((img_x, img_y, 1))
conv1 = Conv_BN_Relu(8, (5,5), 1, input_layer)
conv1_MaxPooling = MaxPooling2D((2,2), padding='same')(conv1)

# conv2_x
x = res_a_or_b(conv1_MaxPooling, 8, 'b')
x = res_a_or_b(x, 8, 'a')

# conv3_x
x = res_a_or_b(x, 16, 'b')
x = res_a_or_b(x, 16, 'a')

# conv4_x
x = res_a_or_b(x, 32, 'b')
x = res_a_or_b(x, 32, 'a')

# conv5_x
x = res_a_or_b(x, 64, 'b')
x = res_a_or_b(x, 64, 'a')

x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(8)(x)
x = Dropout(0.4)(x)
y = Activation('sigmoid')(x)

model = Model([input_layer], [y])

#model.summary()


# binary crossentropy
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# 调试时 以划分出的训练集 训练模型
model.fit(x_train, y_train, batch_size=128, epochs=20)

# 调试时 以划分出的验证集 检验预测准确度
y_predict = model.predict(x_test)
# 为保留模型输出结果，另建数组储存预测结果(0-1)
label_predict = np.zeros((len(y_predict),8), dtype=int)
for i in range(len(y_predict)):
    for j in range(8):
        if y_predict[i][j] > 0.5:
            label_predict[i][j] = 1

# 调试时 计算Hamming distance，考虑到两数组值均为0-1，作差即可区分
distance = np.sum(np.abs(label_predict - y_test))
print("----------------")
print("total Hamming distance on valid sample:",distance/len(y_test)/8)



# 以train_data训练，输出test_data的预测结果
model.fit(train_data, label_train, batch_size=128, epochs=20)

pre_result = model.predict(test_data)
test_label_predict = np.zeros((len(pre_result),8), dtype=int)
for i in range(len(pre_result)):
    for j in range(8):
        if pre_result[i][j] > 0.5:
            test_label_predict[i][j] = 1

df = pd.DataFrame(data=test_label_predict)
df.to_csv('group_7.csv', index=False, header=False)


