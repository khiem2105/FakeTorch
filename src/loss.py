import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss():
    def forward(self, y, yhat):
        return np.square(np.linalg.norm(y - yhat, axis=-1))

    def backward(self, y, yhat):
        return 2 * (yhat - y)