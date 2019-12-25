import numpy as np

class Layer:
    # 層の実装
    def __init__(self, input_size, output_size):
        self.params = {}
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.zeros(output_size)
    
    