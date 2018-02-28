import numpy as np
x = np.array([[0.9, 0.1], [0.5, 0.5]])
print(np.matmul(x, x))
print(np.matmul(np.matmul(x, x), x))
print(np.linalg.matrix_power(x, 5))
print(np.linalg.matrix_power(x, 50))

