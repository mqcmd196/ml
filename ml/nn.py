import numpy as np
from ml.utils import *
import warnings

class Layer(object):
    def __init__(self, optimizer='Momentum SGD', lr=0.01, momentum_alpha=0.9, beta1=0.9, beta2=0.999, weight_decay_rate=5e-4):
        """
        Layerオブジェクトの実装
        このあとの各層に共通する機能を実装する
        """
        self.params = {}
        self.grads = {}
        self.optimizer = optimizer
        self.v = None
        self.h = None
        self.m = None
        self.lr = lr
        self.momentum_alpha = momentum_alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay_rate = weight_decay_rate
        
    def update(self):
        """
        最適化アルゴリズムの実装
        """
        if self.optimizer == 'Momentum SGD':
            if self.v == None:
                self.v = {}
                for key in self.params.keys():
                    self.v[key] = np.zeros(shape=self.params[key].shape, dtype=self.params[key].dtype)
            for key in self.params.keys():
                self.v[key] = self.momentum_alpha * self.v[key] - self.lr * self.grads[key]
                # weight_decay_rateによってパラメータの更新を抑えている
                self.params[key] = (1 - self.lr * self.weight_decay_rate) * self.params[key] + self.v[key] 
        
        elif self.optimizer == 'SGD':
            for key in self.params.keys():
                self.params[key] = self.params[key] - (self.lr + self.grads[key])
                
        elif self.optimizer == 'AdaGrad':
            if self.h == None:
                self.h = {}
                for key in self.params.keys():
                    self.h[key] = np.zeros(shape=self.params[key].shape, dtype=self.params[key].dtype)
            for key in self.params.keys():
                self.h[key] = self.h[key] + self.grads[key] * self.grads[key]
                # 1e-7はzero division防止
                self.params[key] = self.params[key] - (self.lr * self.grads[key] / (np.sqrt(self.h[key]) + 1e-7))

        elif self.optimizer == 'Adam':
            if self.lr > 0.009:
                warnings.warn("The lr value is too big.")
            n_iter = 0
            if self.m == None:
                self.m, self.v = {}, {}
                for key in self.params.keys():
                    self.m[key] = np.zeros(shape=self.params[key].shape, dtype=self.params[key].dtype)
                    self.v[key] = np.zeros(shape=self.params[key].shape, dtype=self.params[key].dtype)

            n_iter += 1
            lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** n_iter) / (1.0 - self.beta1 ** n_iter)

            for key in self.params.keys():
                self.m[key] = self.m[key] + ((1 - self.beta1) * (self.grads[key] - self.m[key]))
                self.v[key] = self.v[key] + ((1 - self.beta2) * (self.grads[key]**2 - self.v[key]))
                self.params[key] = self.params[key] - (lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7))

    def initgrad(self):
        for key in self.params.keys():
            self.grads[key] = np.zeros(shape=self.params[key].shape, dtype=self.params[key].dtype)


class Linear(Layer):
    """
    線形層の実装
    """
    def __init__(self, input_size, output_size, optimizer='Momentum SGD', lr=0.01, momentum_alpha=0.9, beta1=0.9, beta2=0.999, weight_decay_rate=5e-4):
        super().__init__(optimizer, lr, momentum_alpha, beta1, beta2, weight_decay_rate)
        self.params = {}
        self.params['W'] = np.random.normal(scale=np.sqrt(1.0/input_size), size=(input_size, output_size)).astype(np.float32)
        self.params['b'] = np.zeros(output_size)
    
    def forward(self, x):
        self.x = x
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        self.grads['W'] = np.dot(self.x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.params['W'].T)   
    

class ReLU(Layer):
    """
    ReLU層の実装
    """
    def __init__(self, optimizer='Momentum SGD', lr=0.01, momentum_alpha=0.9, beta1=0.9, beta2=0.999, weight_decay_rate=5e-4):
        super().__init__(optimizer, lr, momentum_alpha, beta1, beta2, weight_decay_rate)

    def forward(self, x):
        out = np.maximum(x, 0)
        self._mask = np.sign(out)
        return out

    def backward(self, dout):
        return self._mask * dout


class Sigmoid(Layer):
    """
    Sigmoid層の実装
    """
    def __init__(self, optimizer='Momentum SGD', lr=0.01, momentum_alpha=0.9, beta1=0.9, beta2=0.999, weight_decay_rate=5e-4):
        super().__init__(optimizer, lr, momentum_alpha, beta1, beta2, weight_decay_rate)

    def forward(self, x):
        out = 1 / (1 + np.exp(x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Sequential:
    def __init__(self):
        self.layers = []

    def addlayer(self, layer):
        self.layers.append(layer)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def update(self):
        for layer in self.layers:
            layer.update()

    def initgrad(self):
        for layer in self.layers:
            layer.initgrad()


class Classifier:
    def __init__(self, model):
        self.model = model

    def predict(self, x, t):
        y = self.model.forward(x)
        pred = np.argmax(y, axis=1)
        acc = 1.0 * np.where(pred == t)[0].size / y.shape[0]
        loss = softmax_cross_entropy(y, t)
        return loss, acc

    def update(self, x, t):
        self.model.initgrad()
        y = self.model.forward(x)
        pred = np.argmax(y, axis=1)
        acc = 1.0 * np.where(pred == t)[0].size / y.shape[0]
        prob = softmax(y)
        loss = cross_entropy(prob, t)
        dout = prob
        dout[np.arange(dout.shape[0]), t] -= 1

        self.model.backward(dout / dout.shape[0])
        self.model.update()

        return loss, acc