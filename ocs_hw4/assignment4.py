import numpy as np
import matplotlib.pyplot as plt

import pdb

plt.close('all')
n = 200 # data samples per class
LOAD_DATA = True # if True, always use the same data

m1 = np.array([1,1]) # means
m2 = np.array([5,3.5])
cov1 = np.array([[3,0],[0,0.5]]) # covariances
cov2 = np.array([[1,0.5],[0.5,0.75]])

a = 0.1

def logistic(u):
    """
    Compute logistic function 1 / (1+exp(-x)). x can be a scalar or a vector.
    Avoid numeric overflow.
    """
    u = np.atleast_1d(u)
    idx1 = np.where(u>=0)
    idx2 = np.where(u<0)

    result = np.zeros_like(u)
    result[idx1] = 1.0/(1.0 + np.exp(-u[idx1]))
    result[idx2] = np.exp(u[idx2]) / (1+np.exp(u[idx2]))
    return np.squeeze(result) # in case x was a scalar, return a scalar


def evaluate(p_k, f_star, p_star, list_f):
    list_f.append((f(p_k, x, y) - f_star, np.linalg.norm(p_k - p_star)))

def evaluate_print(list_f):
    print "Format: f(p)-f*, ||pk - p*||"
    for i in list_f:
        print i
    print "-- finished -- \n\n"

def point_7(p, x, y):
    """
    Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    :param p:
    :param x:
    :param y:
    :return: Main Diagonal is True Positives
    """
    list_tmp = list()
    for c, val in enumerate(x.T):
        x_dach = np.append(val, 1.0)
        list_tmp.append((y[c], p.dot(x_dach)))

    confusion = np.zeros((2,2))

    for val in list_tmp:
        if val[0] == 1 and val[1] >= 0:
            confusion[0,0] += 1
        if val[0] == -1 and val[1] < 0:
            confusion[1,1] += 1
        if val[0] == 1 and val[1] < 0:
            confusion[0,1] += 1
        if val[0] == -1 and val[1] >= 0:
            confusion[1,0] += 1

    print confusion

def nabla(p, x, y):
    res1 = np.zeros(3)
    for c, val in enumerate(x.T):
        x_dach = np.append(val, 1.0)
        res1 += (1 - logistic(y[c]*p.reshape((1,3)).dot(x_dach)))*y[c]*x_dach

    return a * S.dot(p) - res1


def nabla_2(x, y):
    sum_w11, sum_w12, sum_w1b, sum_w22, sum_w2b, sum_bb = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for c, val in enumerate(x.T):
        x_dach = np.append(val, 1.0)
        tmp_dev = logistic(y[c]*p.reshape((1,3)).dot(x_dach))
        tmp_dev *= (1-tmp_dev)

        sum_w11 += tmp_dev * x_dach[0]**2
        sum_w12 += tmp_dev * x_dach[0]*x_dach[1]
        sum_w1b += tmp_dev * x_dach[0]
        sum_w22 += tmp_dev * x_dach[1]**2
        sum_w2b += tmp_dev * x_dach[1]
        sum_bb  += tmp_dev

    return np.array([[a + sum_w11, sum_w12, sum_w1b], [sum_w12, a + sum_w22, sum_w2b], [sum_w1b, sum_w2b, sum_bb]])


def f(p, x, y):
    res1 = 0
    for c, val in enumerate(x.T):
        x_dach = np.append(val, 1.0)
        res1 += np.log(logistic(y[c]*p.reshape((1,3)).dot(x_dach)))

    return 0.5*a*np.sum(S.dot(p))-res1


def gradient_checker(x, y, epsilon=0.0001):
    """
    Gradien checker, to know if I dit it right
    :param x: only one value of the set X
    :param y: {-1,1}
    :param epsilon: the smaller epsilon the more precise
    :return: print msg of the conparison. Should be equal.
    """
    def function(p, x, y):
        return a / 2 * np.sum(S.dot(p)**2) - np.log(logistic(y * p.reshape((1, 3)).dot(x)))

    def nabla(p, x, y):
        return a * S.dot(p) - (1 - logistic(y * p.reshape((1, 3)).dot(x))) * y * x

    def nabla_numeric(p, x, y, a_j=0.01):
        nabla = np.zeros(3)
        for i in range(0, len(p)):
            p[i] += epsilon 
            res1 = function(p, x, y)
            p[i] -= 2*epsilon
            res2 = function(p, x, y)
            p[i] += epsilon
            nabla[i] = (res1 - res2) / (2*epsilon)
        return nabla

    x_dach = np.append(x, 1.0)
    print "The numerical nabla   {}\n".format(nabla_numeric(p, x_dach, y))
    print "The analyticlal nabla {}\n".format(nabla(p, x_dach, y))


def steepest_descent(p_k, x, y, eval=False, alpha_n=0.001):
    print "Start the Steeptest Decent Method Jacobian Evaluation with an alpha_n {}.".format(alpha_n)
    i = 0; list_eval = list()
    while True:
        d_k = -nabla(p_k, x, y)
        p_k += alpha_n*d_k

        if (eval):
            evaluate(p_k, f_star, p_star, list_eval)

        # slides gradient methods: p 20
        # print "steepest error: ", np.linalg.norm(nabla(p_k, x, y))/norm_p_0
        if np.linalg.norm(nabla(p_k, x, y))/norm_p_0 <= 0.01:
            point_7(p_k, x, y)
            break

        i += 1
    print "Jacobian Matrix needed {} iterations".format(i)
    evaluate_print(list_eval)

    return p_k


def newton(p_k2, x, y, eval=False, alpha_n2=0.05):
    print "Start the Newton Method Hessian Evaluation with an alpha_n2 {}.".format(alpha_n2)
    i = 0; list_eval = list()
    while True:
        d_k_2 = -(np.linalg.inv(nabla_2(x, y))).dot(nabla(p_k2, x, y))
        p_k2 += alpha_n2 * d_k_2

        if eval:
            evaluate(p_k2, f_star, p_star, list_eval)

        # slides gradient methods: p 20
        # print "newton error: ", np.linalg.norm(nabla(p_k2, x, y))/norm_p_0
        if np.linalg.norm(nabla(p_k2, x, y)) / norm_p_0 <= 0.01:
            print "newton error {}".format(np.linalg.norm(nabla(p_k2, x, y)) / norm_p_0)
            point_7(p_k2, x, y)
            break

        i += 1
    print "Newton needed {} iterations".format(i)
    evaluate_print(list_eval)

    return p_k2

def nesterov(p_k3, x, y, eval=False, alpha_n=0.01):
    print "Start the Nesterov Evaluation with an alpha {}.".format(alpha_n)
    p_prev = np.append(m1, np.random.rand(1))
    t = 0.0; i = 0; list_eval = list()
    while True:
        t_prev = t
        t = 0.5*(1+np.sqrt(1+4*t**2))
        beta = (t_prev-1.0)/t
        q = p_k3 + beta * (p_k3 - p_prev)

        p_prev = p_k3.copy()
        p_k3 = q - alpha_n * nabla(q, x, y)

        if (eval):
            evaluate(p_k3, f_star, p_star, list_eval)

        # print "nesterov error: ", np.linalg.norm(nabla(p_k3, x, y))/norm_p_0
        if np.linalg.norm(nabla(p_k3, x, y)) / norm_p_0 <= 0.01:
            print "nesterov error {}".format(np.linalg.norm(nabla(p_k3, x, y)) / norm_p_0)
            point_7(p_k3, x, y)
            break

        i += 1
    print "Nesterov  needed {} iterations".format(i)
    evaluate_print(list_eval)

    return p_k3

x = np.zeros((2,2*n))
y = np.zeros(2*n)
x[:,0:n] = np.random.multivariate_normal(m1, cov1, n).T
x[:,n:] = np.random.multivariate_normal(m2, cov2, n).T

y[0:n] = 1
y[n:] = -1

x_test = np.zeros((2,2*n))
y_test = np.zeros(2*n)
x_test[:,0:n] = np.random.multivariate_normal(m1, cov1, n).T
x_test[:,n:] = np.random.multivariate_normal(m2, cov2, n).T

y_test[0:n] = 1
y_test[n:] = -1

if LOAD_DATA:
    x = np.load("x.npy")
    y = np.load("y.npy")
    n = x.shape[1]/2
else:
    np.save("x.npy", x)
    np.save("y.npy", y)

S = np.diag((1., 1., 0))
m_1 = np.append(m1, 1)
b = np.random.rand(1)
p = np.append(m1, b)
p_0 = p.copy()
norm_p_0 = np.linalg.norm(nabla(p_0, x, y))

## Newton
# p_steepest = steepest_descent(p.copy(), x, y, alpha_n=0.01)
# p_newton = newton(p.copy(), x, y, alpha_n2=0.01)

# gradient_checker(x[:,0], y[0])

print "------------------------------------------------------"
print " Start Evaluation Environment "
print "------------------------------------------------------"

print "Point 6 - Evaluation of the algorithms with f* and p*"
p_star = p = nesterov(p.copy(), x, y, alpha_n=0.02)
f_star = f(p_star, x, y)
l1 = steepest_descent(p_0.copy(), x, y, eval=True, alpha_n=0.01)
l2 = newton(p_0.copy(), x, y, eval=True, alpha_n2=0.4)
l3 = nesterov(p_0.copy(), x, y, eval=True, alpha_n=0.02)

def plotting(final_p):
    hype_x = np.linspace(np.min(x[0]),  np.max(x[0]), 100)
    hype_y = (final_p[2] + final_p[0] * hype_x) / -final_p[1]

    return hype_x, hype_y

plt.figure()
hype_x, hype_y = plotting(p)
plt.plot(hype_x, hype_y, 'c-', label='nestrov')

hype_x, hype_y = plotting(l1)
plt.plot(hype_x, hype_y, 'm-', label='steeptest')

hype_x, hype_y = plotting(l2)
plt.plot(hype_x, hype_y, 'r-', label='newton')

plt.plot(x[0,0:n], x[1,0:n], 'g+')
plt.plot(x[0,n:], x[1,n:], 'b+')
plt.axis('equal')
plt.legend()
plt.show()
