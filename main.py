from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import Sequential
import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

features = data.drop(['label'], axis=1).values
labels = data['label'].values

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train / 255.
x_test = x_test / 255.

model = Sequential([
    Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'), # 二维卷积，提取64个特征，relu做激活函数
    MaxPooling2D(pool_size=(2, 2)),# 池化，降低维度
    Dropout(0.25), #部分神经元按概率停止前向传播
    Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'), #继续提取32个特征
    MaxPooling2D(pool_size=(2, 2)),#池化
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax'),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test))

def show_example(index, preds, dsx, dsy):
    current_img = dsx[index][:, :, 0] * 255
    prediction = np.argmax(preds[index])
    if (len(dsy) > 0 or dsy != None):
        label = dsy[index]
        print("Label:", label)
    print("Prediction:", prediction)

    plt.imshow(current_img, interpolation='nearest', cmap='gray')
    plt.show()


def analysis(preds, limit, dsx, dsy):
    correct = 0
    misclassified = []
    for i in range(limit):
        prediction = np.argmax(preds[i])
        label = dsy[i]
        if (prediction == label):
            correct += 1
        else:
            misclassified.append(i)

    print(
        f"Predictions in a limit of {limit} are {(correct / limit) * 100} correct")
    print(f"Misclassfied {len(misclassified)} examples:")
    for i in misclassified:
        show_example(i, preds, dsx, dsy)

X_test = pd.read_csv('../input/digit-recognizer/test.csv')
X_test = X_test.to_numpy()
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test = X_test / 255.
preds = model.predict(X_test)
labels = [np.argmax(i) for i in preds]
idxs = [i+1 for i in range(len(labels))]
submit = pd.DataFrame({'ImageId': idxs, 'Label': labels})
submit.to_csv('submission.csv', index=False)