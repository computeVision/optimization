import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import scipy
import scipy.sparse

import cProfile
import re


plt.close('all')

n = 200  # number of data samples
c = np.array([-1, 1])  # center
s = np.array([[0.8, 0.7], [0.7, 0.8]]) * 1.5  # covariance

# for units in (3, 5, 15, 25):
#   print units
#   num_hidden_units = units # number of hidden units
num_hidden_units = 25 # number of hidden units

# generate data samples
t = np.random.multivariate_normal(c, s, n).T  # 2d points
z = np.vstack((t[0, :] * np.cos(t[0, :]), t[1, :], t[0, :] * np.sin(t[0, :])))  # 3d points
plt.figure(); plt.gca(projection='3d').scatter(z[0, :], z[1, :], z[2, :], s=20, c='r')
plt.figure(); plt.plot(t[0, :], t[1, :], 'rx')

# vector with all parameters of the network
# initialize with random values
num_params = 3 * num_hidden_units + num_hidden_units + num_hidden_units * 2 + 2  # total number of parameters in the network
parameters = np.random.randn(num_params) * (1.0 / np.sqrt(num_hidden_units))

def evaluate_f(x, a, b):
    x = np.atleast_2d(x)
    return (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2


def extract_parameters(parameters, num_hidden_units):
    """
    convenience function to extract individual parameter blocks for easy access.
    returns a dictionary with keys 'hidden' and 'output', each of which has a list [W,b]
    of the weight matrix and bias vector for the respective layer
    """
    p = {}
    p['hidden'] = []
    p['hidden'].append(parameters[0:3 * num_hidden_units].reshape(num_hidden_units, 3))
    p['hidden'].append(parameters[3 * num_hidden_units:4 * num_hidden_units])
    p['output'] = []
    p['output'].append(parameters[4 * num_hidden_units:6 * num_hidden_units].reshape(2, num_hidden_units))
    p['output'].append(parameters[6 * num_hidden_units:])

    # print 'hidden ', 'weights ', len(p['hidden'][0]), ' bias ', len(p['hidden'][1])
    # print 'output ', 'weights ', len(p['output'][0]), ' bias ', len(p['output'][1])

    return p


def logistic(x):
    """
    logistic function
    """
    return 1.0 / (1.0 + np.exp(-x))


def grad_logistic(x):
    """
    gradient of logistic function
    """
    sigmoid = logistic(x)
    return sigmoid * (1 - sigmoid)


def compute_layer(x, params, output=False):
    """
    compute the result of a layer for input vector x and layer parameters params.
    If output is True, the layer is an output layer (i.e. does not apply the nonlinear
    activation function)
    """
    weights = params[0]
    bias = params[1]

    y = np.zeros((weights.shape[0],))
    for i in xrange(len(y)):
        y[i] = bias[i] + sum(weights[i] * x)
    if not output:
        y = logistic(y)

    return y


def forward(x, parameters):
    """
    compute a forward pass through the network for input vector x and network parameters
    parameters
    """
    p = extract_parameters(parameters, num_hidden_units=num_hidden_units)
    hidden = compute_layer(x, p['hidden'], output=False)
    output = compute_layer(hidden, p['output'], output=True)

    return output


def compute_loss(x, t, parameters):
    """
    compute the loss for a data sample x (input vector), t (desired output) and network
    parameters parameters
    """
    return 0.5 * sum((forward(x, parameters) - t) ** 2)


def cost_grad(x, t, parameters):
    """
    compute the gradient of the loss function for a data sample x (input vector)
    t (desired output) and network parameters parameters
    """
    p = extract_parameters(parameters, num_hidden_units=num_hidden_units)

    activation = compute_layer(x, p['hidden'], output=True)
    x_tilde = compute_layer(x, p['hidden'], output=False)
    y = compute_layer(x_tilde, p['output'], output=True)

    dev_b2 = y - t
    dev_w2 = np.outer(y - t, x_tilde)

    dev_b1 = grad_logistic(activation) * (y - t).dot(p['output'][0])
    dev_w1 = np.outer(grad_logistic(activation) * ((y - t).dot(p['output'][0])), x)

    return dev_b1, dev_w1, dev_b2, dev_w2


def cost_grad_gauss(x, t, parameters):
    p = extract_parameters(parameters, num_hidden_units=num_hidden_units)

    activation = compute_layer(x, p['hidden'], output=True)
    x_tilde = compute_layer(x, p['hidden'], output=False)

    y = compute_layer(x_tilde, p['output'], output=True)

    dev_b2 = np.eye(2)  # does not change at all
    dev_w2_1 = np.matrix((x_tilde, np.zeros(len(x_tilde)))).T
    dev_w2_2 = np.matrix((np.zeros(len(x_tilde)), x_tilde)).T

    dev_phi = grad_logistic(activation)
    dev_b1 = p['output'][0].T * dev_phi.reshape((len(dev_phi), 1))

    dev_w1 = list()
    x_mat = np.matrix((x, x)).T

    for i in xrange(num_hidden_units):
        dev_w1.append(np.multiply(x_mat, p['output'][0][:, i] * dev_phi[i]))

    return dev_b1, dev_w1, dev_b2, dev_w2_1, dev_w2_2, y - t


###
### network training
###
z_train = z[:, 0:n - 100].copy()  # divide into training and test set
t_train = t[:, 0:n - 100].copy()  # last 100 data samples are the test set
z_test = z[:, n - 100:].copy()
t_test = t[:, n - 100:].copy()

p = parameters.copy()
output_train = np.zeros((t_train.T.shape[0],))

output_train_mean = 0.0
# alpha = 0.01
alpha = 0.003
armijo_alpha = 1.0
x0 = np.array([0, -0.2])  # starting point
x = x0; d = 1; b = 100

lambd = np.array([10. ** -7, 10. ** -4])
simple_algo = False
# simple_algo = True
grad_e_c = np.matrix([[1, 0], [0, 1]])

averagelisttrai = list()
averagelisttest = list()

identity_b1 = scipy.sparse.eye(num_hidden_units, format='csc')
identity_w1 = scipy.sparse.eye(3, format='csc')
identity_b2 = scipy.sparse.eye(2, format='csc')
identity_w2 = scipy.sparse.eye(num_hidden_units, format='csc')

for k in range(2000):
    par = extract_parameters(p, num_hidden_units)
    if simple_algo:
        b1_sum = np.zeros_like(par['hidden'][1])
        w1_sum = np.zeros_like(par['hidden'][0])
        b2_sum = np.zeros_like(par['output'][1])
        w2_sum = np.zeros_like(par['output'][0])
        for index, z in enumerate(z_train.T):
            output_train[index] = compute_loss(z, t_train.T[index], p)
            grad_b1, grad_w1, grad_b2, grad_w2 = cost_grad(z, t_train.T[index], p)
            b1_sum += grad_b1
            w1_sum += grad_w1
            b2_sum += grad_b2
            w2_sum += grad_w2

        b1_sum /= z_train.T.shape[0]
        w1_sum /= z_train.T.shape[0]
        b2_sum /= z_train.T.shape[0]
        w2_sum /= z_train.T.shape[0]

        d = np.hstack((w1_sum.flatten(), b1_sum, w2_sum.flatten(), b2_sum))
        current_output = output_train.mean()

        alpha = 1.0
        stop_crit = False
        for _ in range(25):
            p_ = p - alpha * d
            output_train_sum = 0.0
            for beer, (z, t) in enumerate(zip(z_train.T, t_train.T)):
                output_train_sum += compute_loss(z, t, p_)
            output_train_sum /= z_train.T.shape[0]
            if output_train_sum < current_output:
                # print 'juhu'
                p = p_
                break
            else:
                alpha /= 1.5

        # do not wanna change sign
        if current_output - output_train_sum < 10e-8:
            print 'output_train_sum - current_output < 10e-8'
            break
        if (alpha * d).dot(alpha * d) < 10e-6:
            print '(alpha * d).dot(alpha * d) < 10e-8'
            break

    else: ################################# Gauss Newton ####################################
        d = np.zeros_like(p)
        for i, z in enumerate(z_train.T):
            output_train[i] = compute_loss(z, t_train.T[i], p)
            dev_b1, dev_w1, dev_b2, dev_w2_1, dev_w2_2, error_beer= cost_grad_gauss(z, t_train.T[i], p)

            nabla = np.vstack(tuple(map(lambda x: np.array(x), dev_w1))+(np.array(dev_b1), np.array(dev_w2_1), np.array(dev_w2_2), np.array(dev_b2)))
            d += -np.linalg.inv(lambd[0] * np.eye(len(p)) + np.dot(nabla, nabla.T)).dot(nabla.dot(error_beer))
        d /= z_train.T.shape[0]

        current_output = output_train.mean()
        alpha = 5.0
        stop_crit = False
        for _ in range(200):
            p_ = p + alpha * d
            output_train_sum = 0.0
            for index, (z, t) in enumerate(zip(z_train.T, t_train.T)):
                output_train_sum += compute_loss(z, t, p_)
            output_train_sum /= z_train.T.shape[0]
            if output_train_sum < current_output:
                # print 'juhu'
                p = p_
                break
            else:
                alpha /= 1.5

        if current_output - output_train_sum < 10e-7:
            print 'current_output - output_train_sum'
            break

        #euclidian distance
        if (alpha * d).dot(alpha * d) < 10e-5:
            print '(alpha * d).dot(alpha * d)'
            break

    output_test_sum = 0.0
    for aa, (zi, ti) in enumerate(zip(z_test.T, t_test.T)):
        output_test_sum += compute_loss(zi, ti, p_)
    output_test_sum /= z_test.T.shape[0]
    print 'k=', k, 'current_output=', current_output, 'test=', output_test_sum, 'alpha=', alpha
    averagelisttrai.append(current_output)
    averagelisttest.append(output_test_sum)

print 'mean train ', reduce(lambda x, y: x + y, averagelisttrai) / len(averagelisttrai)
print 'mean test  ', reduce(lambda x, y: x + y, averagelisttest) / len(averagelisttest)

### evaluate on test set ###
learned_parameters = p.copy()
y_init = np.zeros((2, z_test.shape[1]))
y_learned = np.zeros_like(y_init)
for idx, x in enumerate(z_test.T):
    y_init[:, idx] = forward(x, parameters)  # compute output of network with initial parameters
    y_learned[:, idx] = forward(x, learned_parameters)  # output with learned parameters

plt.figure();
plt.title('test set')
plt.plot(y_init[0, :], y_init[1, :], 'rx')
plt.plot(y_learned[0, :], y_learned[1, :], 'bx')
plt.plot(t_test[0, :], t_test[1, :], 'gx')
plt.legend(('initial parameters', 'learned parameters', 'groundtruth'))

# if (simple_algo):
#     plt.savefig('report/simple_algo/' + 'test_' + str(num_hidden_units) + '.png')
# else:
#     plt.savefig('report/compli_algo/' + 'test_' + str(num_hidden_units) + '.png')
#
for idx, x in enumerate(z_train.T):
    y_init[:, idx] = forward(x, parameters)  # compute output of network with initial parameters
    y_learned[:, idx] = forward(x, learned_parameters)  # output with learned parameters

plt.figure();
plt.title('training set')
plt.plot(y_init[0, :], y_init[1, :], 'rx')
plt.plot(y_learned[0, :], y_learned[1, :], 'bx')
plt.plot(t_test[0, :], t_test[1, :], 'gx')
plt.legend(('initial parameters', 'learned parameters', 'groundtruth'))

# if (simple_algo):
#     plt.savefig('report/simple_algo/' + 'train_' + str(num_hidden_units) + '.png')
# else:
#     plt.savefig('report/compli_algo/' + 'train_' + str(num_hidden_units) + '.png')

plt.show()
