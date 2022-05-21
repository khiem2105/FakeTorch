from calendar import c
from turtle import forward
import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, delta):
        ## Calcul la derivee de l'erreur
        pass

class Linear(Module):
    def __init__(self, d_prev, d_next, init="xavier", grad_norm=None):
        super(Linear, self).__init__()
        self._gradient = np.zeros((d_prev, d_next))
        self._bias_gradient = np.zeros((1, d_next))
        
        # weights initilization
        if init == "xavier":
            self._parameters = np.random.uniform(low=-1., high=1., size=(d_prev, d_next)) * np.sqrt(6. / (d_prev + d_next))
        else:
            self._parameters = np.random.uniform(low=-1., high=1., size=(d_prev, d_next))

        # bias initialization
        self._bias = np.zeros((1, d_next))

        self.dim = (d_prev, d_next)

        self.grad_norm = grad_norm

    def zero_grad(self):
        self._gradient = np.zeros(self.dim)
        self._bias_gradient = np.zeros((1, self.dim[1]))

    def forward(self, X):
        assert X.shape[1] == self.dim[0], "input not in right shape"
        self.input = X

        # Z_h = Z_{h-1} * W_h + bias
        return np.dot(X, self._parameters) + self._bias

    def update_parameters(self, gradient_step=0.001):
        self._parameters = self._parameters - self._gradient * gradient_step
        self._bias = self._bias - self._bias_gradient * gradient_step

    def backward_update_gradient(self, delta):
        assert delta.shape[1] == self.dim[1], "delta not in right shape"

        # delta_W = X^T * delta
        self._gradient = self._gradient + np.dot(self.input.T, delta)
        self._bias_gradient = self._bias_gradient + np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, delta):
        assert delta.shape[1] == self.dim[1], "delta not in right shape"
        
        # delta_prev = delta * W^T
        return np.dot(delta, self._parameters.T)

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.output = None
    
    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))

        return self.output

    def update_parameters(self, gradient_step=0.001):
        pass

    def backward_delta(self, delta):
        return delta * self.output * (1 - self.output)


class TanH(Module):
    def __init__(self):
        super(TanH, self).__init__()
        self.output = None
    
    def forward(self, X):
        self.output = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

        return self.output

    def update_parameters(self, gradient_step=0.001):
        pass

    def backward_delta(self, delta):
        return delta * (1 - np.square(self.output))

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.output = None
    
    def forward(self, X):
        self.X = X
        self.output = X > 0 * X

        return self.output

    def update_parameters(self, gradient_step=0.001):
        pass

    def backward_delta(self, delta):
        return delta * self.X > 0


class Sequentiel(object):
    def __init__(self, *layers):
        super(Sequentiel, self).__init__()
        self.layers = [*layers]

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def backward(self, delta):
        for layer in self.layers[::-1]:
            layer.backward_update_gradient(delta)
            delta = layer.backward_delta(delta)
