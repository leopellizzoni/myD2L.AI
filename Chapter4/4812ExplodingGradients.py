from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('after multiplying 100 matrices', M)