from mxnet import np, npx
npx.set_np()

# A = np.arange(9).reshape(3,3)
# print(A)
#B = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 5]])
# print(B)

# x = np.array([1, 1, 2, 3])
# y = np.array([1, 1, 2, 3])
# print(np.vdot(x, y))

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

# for i in range(8):
#     print(f'{i} --> {f(i)}')

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1