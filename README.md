# DigitRecognition
使用CNN实现的数字识别。kaggle 比赛地址：https://www.kaggle.com/competitions/digit-recognizer

CNN 学习地址：https://blog.csdn.net/sinat_29957455/article/details/78289987

源代码位于 ```main.py```，训练集（训练组和测试组）为 ```test.csv```，测试集为 ```test.csv```。代码中文件路径为 kaggle 中的文件路径。

首先将用 ```train_test_split``` 输入数据分成两组：训练组和测试组。

使用 CNN 对数据进行处理：

1. 用二维卷积提取图中的 64 个特征，得到 $28 \times 28 \times 32$ 的矩阵。对数据进行池化降低维度，并使用 dropout 层防止过拟合。
2. 对第一次处理后的数据再提取 32 个特征，得到 $14 \times 14 \times 32$ 的矩阵。继续对数据池化并使用 dropout。
3. 使用两层 Dense 对特征进行分析和处理，最后使用 argmax 函数输出最可能的预测结果。
