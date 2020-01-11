# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from ml.nn import *
from datasets.dataset import load_mnist
import matplotlib.pyplot as plt


# %%
(x_train, t_train), (x_test, t_test) = load_mnist()


# %%
img = np.reshape(x_train[1], (28, 28))


# %%
input_size = 784
output_size = 10
middle_layer_size = 50
optimizer = 'momentum SGD'


# %%
model = Sequential()
model.addlayer(Linear(input_size, middle_layer_size, optimizer=optimizer))
model.addlayer(ReLU(optimizer=optimizer))
model.addlayer(Linear(middle_layer_size, middle_layer_size, optimizer=optimizer))
model.addlayer(ReLU(optimizer=optimizer))
model.addlayer(Linear(middle_layer_size, output_size, optimizer=optimizer))
network = Classifier(model)


# %%
batch_size = 100
epoch = 10
n_train = x_train.shape[0]
n_test = x_test.shape[0]


# %%
for e in range(epoch):
    print('epoch %d'%e)
    randinds = np.random.permutation(n_train)
    for it in range(0, n_train, batch_size):
        ind = randinds[it:it+batch_size]
        x = x_train[ind]
        t = t_train[ind]
        loss, acc = network.update(x, t)
        print('train iteration %d, loss %f, acc %f'%(it//batchsize, loss, acc))

    acctest = 0
    losstest = 0
    for it in range(0, n_test, batchsize):
        x = x_test[it:it+batchsize]
        t = t_test[it:it+batchsize]
        loss, acc = network.predict(x, t)
        acctest += int(acc * batchsize)
        losstest += loss
    acctest /= (1.0 * n_test)
    losstest /= (n_test // batchsize)
    print('test, loss %f, acc %f'%(loss, acc))


# %%
x.shape

