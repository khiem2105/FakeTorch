import numpy as np

from numba import jit

class Optim(object):
    def __init__(self, net, loss, lr=1e-3):
        super(Optim, self).__init__()
        self.net = net
        self.loss = loss
        self.lr = lr

    def step(self, batch_x, batch_y, epoch):
        yhat = self.net.forward(batch_x)

        delta = self.loss.backward(batch_y, yhat)
        delta = delta / delta.shape[0]
        self.net.backward(delta)

        for layer in self.net.layers:
            layer.update_parameters(gradient_step=self.lr)

    def zero_grad(self):
        for layer in self.net.layers:
            layer.zero_grad()

class SGD(Optim):
    def __init__(self, net, loss, batch_size=32, lr=1e-3, shuffle=True):
        super(SGD, self).__init__(net, loss, lr)
        self.batch_size = batch_size
        self.shuffle = shuffle
    

    def iterate_mini_batch(self, data, label):
        if self.shuffle:
            ind = np.arange(data.shape[0])
            np.random.shuffle(ind)

        for start_ind in range(0, data.shape[0] - self.batch_size + 1, self.batch_size):
            if self.shuffle:
                batch_ind = ind[start_ind:start_ind+self.batch_size]
            else:
                batch_ind = slice(start_ind, start_ind+self.batch_size)
            
            yield data[batch_ind], label[batch_ind]

class Adam(SGD):
    def __init__(self, net, loss, batch_size=32, lr=1e-3, beta1=.9, beta2=.99, eps=1e-8, shuffle=True):
        super(Adam, self).__init__(net, loss, batch_size, lr, shuffle)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    def step(self, batch_x, batch_y, epoch):
        yhat = self.net.forward(batch_x)

        delta = self.loss.backward(batch_y, yhat)
        delta = delta / delta.shape[0]
        self.net.backward(delta, adaptative=True, beta1=self.beta1, beta2=self.beta2)

        for layer in self.net.layers:
            layer.update_parameters(gradient_step=self.lr, adaptative=True, beta1=self.beta1, beta2=self.beta2, eps=self.eps, epoch=epoch)
