import numpy as np

def softmax(y):
    y = y - np.max(y, axis=1, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
    
def cross_entropy(prob, t):
    if prob.ndim == 1:
        t = t.reshape(1, t.size)
        prob = prob.reshape(1, prob.size)
    return -np.mean(np.log(prob[np.arange(prob.shape[0]), t] + 1e-7))

def softmax_cross_entropy(y, t):
    return cross_entropy(softmax(y), t)