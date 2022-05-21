from telnetlib import SGA
import numpy as np

class Optim(object):
    def __init__(self, net, loss, lr=1e-3):
        super(Optim, self).__init__()
        self.net = net
        self.loss = loss
        self.lr = lr

    def step(self, batch_x, batch_y):
        yhat = self.net.forward(batch_x)

        delta = self.loss.backward(batch_y, yhat)
        delta = delta / delta.shape[0]
        self.net.backward(delta)

        for layer in self.net.layers:
            layer.update_parameters(gradient_step=self.lr)

    def zero_grad(self):
        for layer in self.net.layers:
            layer.zero_grad()


class SGD(object):
    def __init__(self, net, loss, data, label, batch_size=32, epoch=10, lr=1e-3, shuffle=True):
        assert data.shape[0] == label.shape[0], "Data and label must have the same batch size"
        super(SGD, self).__init__()
        self.net = net
        self.loss = loss
        self.lr = lr
        self.epoch = epoch
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def iterate_mini_batch(self):
        if self.shuffle:
            ind = np.arange(self.data.shape[0])
            np.random.shuffle(ind)
        
        for start_ind in range(0, self.data.shape[0] - self.batch_size + 1, self.batch_size):
            if self.shuffle:
                batch_ind = ind[start_ind:start_ind+self.batch_size]
            else:
                batch_ind = slice(start_ind, start_ind+self.batch_size)
            
            yield self.data[batch_ind], self.label[batch_ind]

    def run(self):
        for _ in range(self.epoch):
            for batch in self.iterate_mini_batch():
                batch_x, batch_y = batch

                yhat = self.net.forward(batch_x)

                delta = self.loss.backward(batch_y, yhat)
                delta = delta / delta.shape[0]
                self.net.backward(delta)

                for layer in self.net.layers:
                    layer.update_parameters(gradient_step=self.lr)
                    layer.zero_grad()
