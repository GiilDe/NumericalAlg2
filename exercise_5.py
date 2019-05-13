import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, norm

###########
#PART 1
###########

data = pd.read_csv('data.csv')
func_orig = data.to_numpy()
func = func_orig.copy()

for row in func:
    if 24 <= row[0] <= 54:
        row[1] += np.random.normal(0, 0.1)
    else:
        row[1] += np.random.normal(0, 0.5)

b = func[:, 1] #=y
b_orig = func_orig[:, 1]
X = func[:, 0]

plt.plot(X, b, label='noisy')
plt.plot(X, b_orig, label='original')
plt.title("original function to noisy function")
plt.legend()
plt.show()

###########
#PART 2
###########

m = func.shape[0] #num of samples
d = 15

X_list = X.reshape((m, 1))

ones_vec = np.ones(shape=(m, 1))
A = np.append(ones_vec, X_list, axis=1)

for power in range(2, d+1):
    x_by_power = np.power(A[:, -1], power).reshape((m, 1))
    A = np.append(A, x_by_power, axis=1)


x, _, _, _ = lstsq(A, b, rcond=None)

y_hat = A@x

plt.plot(X, b, label='noisy')
plt.plot(X, b_orig, label='original')
plt.plot(X, y_hat, label='approximated')

plt.legend()
plt.title("original function to noisy function to approximated")
plt.show()

###########
#PART 3
###########

weights = np.ones(shape=(m, 1))
weights[24:55] *= 100

W = np.zeros(shape=(m, m))

W[range(m), range(m)] = weights[:, 0]

x_weighted, _, _, _ = lstsq(A.transpose()@W@A, A.transpose()@W@b, rcond=None)

y_hat_weighted = A@x_weighted

plt.plot(X, b, label='noisy')
plt.plot(X, b_orig, label='original')
plt.plot(X, y_hat, label='approximated')
plt.plot(X, y_hat_weighted, label='approximated weighted')

plt.legend()
plt.title("original function to noisy function to weighted approximation")
plt.show()

###########
#PART 4
###########

d = np.zeros(m)
C = A.copy()
C = np.delete(C, -1, axis=1)
C = np.append(np.zeros(shape=(m, 1)), C, axis=1)
C *= list(range(16))
lambdas = np.linspace(start=0, stop=0.5, num=101)
errors = []

for lambda_ in iter(lambdas):
    first = A.transpose()@A + lambda_*C.transpose()@C
    second = A.transpose()@b + lambda_*C.transpose()@d
    x_reg, _, _, _ = lstsq(first, second, rcond=None)
    y_hat_reg = A@x_reg
    errors.append((norm(y_hat_reg-b_orig), y_hat_reg, lambda_))

plt.plot(lambdas, [error[0] for error in errors])
plt.title("Error as a function of lambda")
plt.show()

best_error, y_hat_reg, best_lambda = min(errors, key=lambda x: x[0])
print("The best lambda is", best_lambda, "with an error of", best_error,"\n")

plt.plot(X, b, label='noisy')
plt.plot(X, b_orig, label='original')
plt.plot(X, y_hat, label='approximated')
plt.plot(X, y_hat_weighted, label='approximated weighted')
plt.plot(X, y_hat_reg, label='best lambda')

plt.legend()
plt.title("everyone together as an happy family")
plt.show()

###########
#PART 5
###########

err_LS = norm(b_orig - y_hat)
err_WLS = norm(b_orig - y_hat_weighted)
err_reg_opt = norm(b_orig - y_hat_reg)
err_WLS_to_LS = err_WLS/err_LS
err_reg_to_LS = err_reg_opt/err_LS
err_0_dot_1 = norm(y_hat[24:55] - b_orig[24:55])
err_WLS_0_dot_1 = norm(y_hat_weighted[24:55] - b_orig[24:55])
err_WLS_to_LS_0_dot_1 = err_WLS_0_dot_1/err_0_dot_1

print("the best polynom is the one with regulation (the socialists were right) with an error of", err_reg_opt,
      "then the standart LS and then the WLS\n")

print("we can see that the ratio err_WLS_to_LS_0_dot_1 is less than 1 meaning that the error of WLS on the part where "
      "we raised the weights to 100 is indeed lower than the error of LS as expected\n")
