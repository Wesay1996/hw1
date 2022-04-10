import numpy as np
import struct
import matplotlib.pyplot as plt


class ANN:
    def __init__(self, hiddenlayerSize, learning_rate, regularization_factor, trainSet, testSet):
        self.dataNum = len(trainSet)                # 训练集样本数
        self.trainSet = trainSet                    # 训练集
        self.lrate = learning_rate                  # 学习率
        self.reg_factor = regularization_factor     # 正则化系数
        self.hiddenlayerSize = hiddenlayerSize      # 隐藏层神经元个数
        self.batchSize = 3000
        self.loss = []                              # 损失函数
        self.val_loss = []
        self.val_accuracy = []
        self.testSet = testSet

        self.Win = (np.random.rand(28 * 28, self.hiddenlayerSize) - 0.5) * 2/ 28                            # 输入层、隐藏层权重
        self.bin = np.zeros((1, self.hiddenlayerSize))                                                      # 输入层、隐藏层偏置
        self.Wout = (np.random.rand(self.hiddenlayerSize, 10) - 0.5) * 2 / np.sqrt(self.hiddenlayerSize)    # 隐藏层、输出层权重
        self.bout = np.zeros((1, 10))                                                                       # 隐藏层、输出层偏置
    
    def train(self, epoch):
        iterNum = self.dataNum//self.batchSize
        for _ in range(epoch):
            #1 打乱训练集
            np.random.shuffle(self.trainSet)
            for i in range(iterNum):
                #print('epoch ' + str(_) + ' iteration ' + str(i+1))
                images = self.trainSet[i * self.batchSize: (i+1) * self.batchSize, :-1]
                labels = self.trainSet[i * self.batchSize: (i+1) * self.batchSize, -1:]
                self.update(images, labels)
                images1 = self.testSet[i * self.batchSize: (i + 1) * self.batchSize, :-1]
                labels1 = self.testSet[i * self.batchSize: (i + 1) * self.batchSize, -1:]
                self.predict(testdata, testlabel)

    
    def update(self, data, labels):
        '''正向传播'''
        # 隐藏层
        hiddenlayerOutput = np.maximum(np.matmul(data, self.Win) + self.bin, 0)
        outlayer = np.maximum(np.matmul(hiddenlayerOutput, self.Wout) + self.bout, 0)

        # Softmax分类器
        scores = np.exp(outlayer)  # self.batchSize * 10
        scoresSum = np.sum(scores, axis=1, keepdims=True)  # self.batchSize * 1
        # classification_prob = scores / scoresSum
        
        # 损失函数选择交叉熵
        temp = np.empty((self.batchSize, 1))
        for i in range(self.batchSize):
            temp[i] = scores[i][int(labels[i])]/scoresSum[i]
        crossEntropy = - np.log(temp)

        # 损失函数 = 交叉熵损失项 + L2正则化项
        loss = np.mean(crossEntropy, axis=0)[0] + 0.5 * self.reg_factor * (np.sum( self.Win * self.Win) + np.sum(self.Wout * self.Wout))
        print('loss is:')
        print(loss)
        self.loss.append(loss)  # 储存每个batch训练的损失大小

        '''反向传播'''

        # 最后一层残差 res 交叉熵损失取softmax激活函数
        res = scores / scoresSum  # self.batchSize * 10
        for i in range(self.batchSize):
            res[i][int(labels[i])] -= 1
        res  /= self.batchSize
        
        dWout = np.matmul( hiddenlayerOutput.T, res)
        dbout = np.sum(res, axis=0, keepdims=True)

        dh = np.dot( res, self.Wout.T) # batchsize * self.hiddenlayerSize
        dh[hiddenlayerOutput<=0] = 0 # Relu求导的结果

        dWin = np.dot( data.T, dh)
        dbin = np.sum( dh, axis=0, keepdims=True)

        # L2正则化求导项
        dWout += self.reg_factor * self.Wout
        dWin += self.reg_factor * self.Win

        # 更新参数
        self.Wout += -self.lrate * dWout
        self.Win += -self.lrate * dWin
        self.bout += -self.lrate * dbout
        self.bin += -self.lrate * dbin
        
        return
    
    def predict(self, testdata, testlabel):
        hiddenlayerOutput = np.maximum(np.matmul(testdata, self.Win) + self.bin, 0)
        outlayer = np.maximum(np.matmul(hiddenlayerOutput, self.Wout) + self.bout, 0)
        prediction = np.argmax(outlayer, axis=1).reshape((len(testdata),1))
        accuracy = np.mean(prediction == testlabel)
        print('测试集准确率为' + str(accuracy))
        self.val_accuracy.append(accuracy)

    def visualization(self):
        # 添加子图
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.plot(self.loss, 'r')
        plt.tight_layout()
        ax2.plot(self.val_accuracy, 'b')
        plt.tight_layout()
        plt.show()




''' 导入图像数据 '''
# 获取图像数据
def images_data(dtype):
    if dtype == 'train':    # 训练集
        data = open('train-images.idx3-ubyte', 'rb').read()
    else:                   # 测试集
        data = open('t10k-images.idx3-ubyte', 'rb').read()
    index = 0
    fmt_header = '>IIII'
    magicnum, imagenum, rownum, colnum = struct.unpack_from(fmt_header, data, index)
    index += struct.calcsize('>IIII')  

    output = np.empty((imagenum, rownum * colnum))
    # 图像数据28*28
    fmt_image = '>' + str(rownum * colnum) + 'B'
    for i in range(imagenum):
        output[i] = np.array(struct.unpack_from(fmt_image, data, index)).reshape((1, rownum * colnum))
        index += struct.calcsize(fmt_image)
    return output


# 获取图像标签
def labels_data(dtype):
    if dtype == 'train':
        data = open('train-labels.idx1-ubyte', 'rb').read()
    else:
        data = open('t10k-labels.idx1-ubyte', 'rb').read()
    index = 0
    fmt_header = '>II'
    magicnum, labelnum = struct.unpack_from(fmt_header, data, index)
    index += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty((labelnum, 1))
    for i in range(labelnum):
        labels[i] = np.array(struct.unpack_from(fmt_image, data, index)[0]).reshape((1, 1))
        index += struct.calcsize(fmt_image)
    return labels



if __name__ == "__main__":


    trainSet = images_data('train')
    trainlabel = labels_data('train')
    trainSet = np.append(trainSet, trainlabel, axis=1)

    testdata = images_data('test')
    testlabel = labels_data('test')
    testSet = np.append(testdata, testlabel, axis=1)
    model = ANN(1000, 0.001, 0.1, trainSet, testSet)

    model.train(50)

    model.predict(testdata, testlabel)
    model.visualization()
