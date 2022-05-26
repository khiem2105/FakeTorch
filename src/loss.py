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
        # return y * np.log(1 + np.exp(-yhat)) + (1 - y) * np.log(1 + np.exp(yhat))
        return -y * np.log(yhat + 1e-8) - (1 - y) * np.log(1 - yhat + 1e-8)

    def backward(self, y, yhat):
        # return 1 / (1 + np.exp(-yhat)) - y
        return -y / (yhat + 1e-8) + (1 -y) / (1 - yhat + 1e-8)

class CrossEntropy():
    def forward(self, y, yhat):
        exp = np.exp(yhat)
        softmax = exp / np.sum(exp, axis=1, keepdims=True)

        return -np.sum(y * np.log(softmax), axis=1, keepdims=True)

    def backward(self, y, yhat):
        exp = np.exp(yhat)
        softmax = exp / np.sum(exp, axis=1, keepdims=True)

        return softmax - y