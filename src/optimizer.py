import numpy as np

class Optim(object):
    def __init__(self, net, loss, lr=1e-3):
        super(Optim, self).__init__()
        self.net = net
        self.loss = loss
        self.lr = lr
        self.train_loss = []

    def step(self, batch_x, batch_y):
        yhat = self.net.forward(batch_x)
        self.train_loss.append(np.sum(self.loss.forward(batch_y, yhat)))

        delta = self.loss.backward(batch_y, yhat)
        self.net.backward(delta)

        for layer in self.net.layers:
            layer.update_parameters(gradient_step=self.lr)