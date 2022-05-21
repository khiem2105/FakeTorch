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

class BinaryCrossEntropyLoss():
    # Combine Sigmoid in BinaryCrossEntropyLoss
    def forward(self, y, yhat):
        return y * np.log(1 + np.exp(-yhat)) + (1 - y) * np.log(1 + np.exp(yhat))
        # return -y * np.log(yhat) - (1 - y) * np.log(1 - yhat)

    def backward(self, y, yhat):
        return 1 / (1 + np.exp(-yhat)) - y
        # return (yhat - y) / (yhat * (1 - yhat))

class CrossEntropy():
    def forward(self, y, yhat):
        exp = np.exp(yhat)
        softmax = exp / np.sum(exp)

        return np.dot(y.T, softmax)

    def backward(self, y, yhat):
        return np.exp(yhat) / np.sum(np.exp(yhat)) - y