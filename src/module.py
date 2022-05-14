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

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

class Linear(Module):
    def __init__(self, d_prev, d_next, init="xavier"):
        self._gradient = np.zeros((d_prev, d_next))
        
        if init == "xavier":
            self._parameters = np.random.uniform(low=-1., high=1., size=(d_prev, d_next)) * np.sqrt(6. / (d_prev + d_next))
        else:
            self._parameters = np.random.uniform(low=-1., high=1., size=(d_prev, d_next))

        self.dim = (d_prev, d_next)

    def zero_grad(self):
        self._gradient = np.zeros(self.dim)

    def forward(self, X):
        assert X.shape[1] == self.dim[0], "X not in right shape"

        # Z_h = Z_{h-1} * W_h
        return np.dot(X, self._parameters)

    def update_parameters(self, gradient_step=0.001):
        self._parameters = self._parameters - self._gradient * gradient_step

    def backward_update_gradient(self, X, delta):
        assert X.shape[1] == self.dim[0], "X not in right shape"
        assert delta.shape[1] == self.dim[1], "delta not in right shape"

        # delta_W = X^T * delta
        self._gradient = self._gradient + np.dot(X.T, delta)

    def backward_delta(self, X, delta):
        assert X.shape[1] == self.dim[0], "X not in right shape"
        assert delta.shape[1] == self.dim[1], "delta not in right shape"
        
        # delta_prev = delta * W^T
        return np.dot(delta, self._parameters.T)