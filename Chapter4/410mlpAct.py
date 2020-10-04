from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

##4.1.2 Activation Functions
##rectified linear unit (ReLU) function
x = np.arange(-8.0, 8.0, 0.1)

print(x)

x.attach_grad()
with autograd.record():
    y = npx.relu(x)

# print(y);
# d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
# d2l.plt.show()

##Sigmoid
with autograd.record():
    y = npx.sigmoid(x)
# print(y)
# d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
# d2l.plt.show()

###tanh - tangent hyperbolic
with autograd.record():
    y = np.tanh(x)
print(y)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
d2l.plt.show()