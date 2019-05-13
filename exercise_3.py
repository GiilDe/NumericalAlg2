import numpy as np

A = np.array([[1/2,1,0,0,1,0],
              [0,0,1/2,1,0,1],
              [3/2,1,0,0,1,0],
              [0,0,3/2,1,0,1],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1],
              [1.2,0,0,0,1,0],
              [0,0,1.2,0,0,1]])

b = np.array([0,1,1,1,0,0,1,0])
x,_,_,_ = np.linalg.lstsq(A, b)
temp1 = A.transpose()@A@x
temp2 = A.transpose()@b
x[2] = 0
x[5] = 0
x[3] = 1
res = A@x