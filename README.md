# hw1
首先随机初始化隐藏层和输出层的网络参数w和b，优化器选择随机梯度下降法（SGD），其中训练的每一个batch_size设置为50，
每一轮epoch的迭代次数即训练集的行数bacth_size，采用交叉熵损失函数及l2正则化来计算每一次迭代过程中的损失，参数的更新我们采用反向传播算法，最终训练出模型。

#测试步骤
共有三个超参数可供进行模型调参，分别为学习率,隐藏层神经元个数，正则化强度，为了减少调参规模，
首先初始化隐藏层 = 100, 正则化强度 = 0.01, 学习率分别选择0.1，0.01，0.001，0.0001对模型进行训练并得到测试集预测准确率
发现学习率为0.001时模型准确度最高
接着调试隐藏层神经元个数和正则化强度，我们分别将隐藏层神经元个数取50，100，500，1000，正则化强度分别取0.1，0.01，0.001训练模型并预测测试集准确率，
发现隐藏层神经元个数为1000，正则化强度为0.1时，模型准确度最高.