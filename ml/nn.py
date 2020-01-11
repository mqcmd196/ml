import numpy as np

class Layer(object):
    def __init__(self, lr=0.01, momentum_alpha=0.9, optimizer='momentum SGD'):
        """
        Layerオブジェクトの実装
        このあとの各層に共通する機能を実装する
        """
        self.params = {}
        self.grads = {}
        self.v = None
        self.h = None
        self.momentum_alpha = momentum_alpha
        self.lr = lr
        self.optimizer = optimizer

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
                # weight_decay_rateによってパラメータの更新に罰則を抑えている
                self.params[key] = (1 - self.lr * self.weight_decay_rate) * self.params[key] + self.v[key] 
        
        elif self.optimizer == 'SGD':
            for key in self.params.keys():
                self.params[key] -= self.lr + grads[key]

        elif self.optimizer == 'AdaGrad':
            if self.h = None:
                self.h = {}
                for key in self.params.keys():
                    self.h[key] = np.zeros(shape=self.params[key].shape, dtype=self.params[key].dtype)
            for key in self.params.keys():
                self.h[key] += self.grads[key] ** 2
                self.params[key] -= self.lr * self.grads[key] / (np.sqrt(self.h[key]) + 1e-7)

        else:
            raise ValueError("The optimizer is not defined.")

    def initgrad(self):
        for key in self.params.keys():
            self.grads[key] = np.zeros(shape=self.params[key].shape, dtype=self.params[key].dtype)


class LinearLayer(Layer):
    """
    線形層の実装
    """
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.params = {}
        self.params['W'] = np.random.randn(input_size, output_size) # 標準正規分布に従う乱数
        self.params['b'] = np.zeros(output_size)
    
    def forward(self, x):
        self.x = x
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        self.grads['W'] = np.dot(self.x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.params['W'].T)   
    


    