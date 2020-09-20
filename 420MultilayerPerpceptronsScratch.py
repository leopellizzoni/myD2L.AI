from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#4.2.1
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()

#4.2.2
def relu(X):
    return np.maximum(X, 0)

#4.2.3
####Because we are disregarding spatial structure, we reshape each two-dimensional image into a flat vector of length num_inputs
def net(X):
    X = X.reshape((-1, num_inputs)) 
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2

#4.2.4
loss = gluon.loss.SoftmaxCrossEntropyLoss()

#4.2.5
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))


d2l.predict_ch3(net, test_iter) #evaluate model with test data

d2l.plt.show()

