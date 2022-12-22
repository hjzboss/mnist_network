import numpy as np
from dataset.mnist import load_mnist
from network import TwoLayerNet
import matplotlib.pyplot as plt

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)  # 获取数据
    batch_size = 100
    iters_num = 10000
    train_size = x_train.shape[0]
    learning_rate = 0.1
    train_loss_list = []
    test_acc_list = []
    train_acc_list = []
    iters = 0
    iter_per_epoch = max(train_size / batch_size, 1)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = network.gradient(x_batch, t_batch)
        for key in ('w1', 'b1', 'w2', 'b2'):
            network.params[key] -= learning_rate * grads[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            iters += 1
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)

    x1 = range(iters_num)
    x2 = range(1, iters + 1, 1)
    plt.subplot(1, 2, 1)
    plt.plot(x1, train_loss_list)
    plt.xlabel('iters')
    plt.ylabel('train_loss')
    plt.subplot(1, 2, 2)
    plt.plot(x2, train_acc_list, color='g', linestyle='-', label='train_acc')
    plt.plot(x2, test_acc_list, color='r', linestyle='-', label='test_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
