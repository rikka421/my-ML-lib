import numpy as np

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[1], [2]])

print(np.sum(arr1, axis=0))
# print(arr1.shape, arr2.shape)