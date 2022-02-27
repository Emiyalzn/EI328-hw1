import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_prime(x):
    fx = sigmoid(x)
    return fx * (1. - fx)

class MLP:
    def __init__(self, layers_dim):
        self.num_layers = len(layers_dim)
        self.parameters = {}
        self.gradients = {}
        self.moments = {}
        self.x_hats = {}
        self.xs = {}
        for i in range(1, self.num_layers):
            self.parameters["w" + str(i)] = np.random.normal(size=(layers_dim[i], layers_dim[i - 1]))
            self.parameters["b"+str(i)] = np.random.normal(size=(layers_dim[i],1))
            self.moments["w"+str(i)] = np.zeros((layers_dim[i], layers_dim[i-1]))
            self.moments["b"+str(i)] = np.zeros((layers_dim[i],1))

    def forward(self, x):
        x = x[:, np.newaxis]
        self.x_hats["layer0"] = x # non-sense
        self.xs["layer0"] = x
        for i in range(1, self.num_layers):
            x_hat = self.parameters["w"+str(i)].dot(x) + self.parameters["b"+str(i)]
            x = sigmoid(x_hat)
            self.x_hats["layer"+str(i)] = x_hat
            self.xs["layer"+str(i)] = x
        return x

    def backward(self, y):
        self.gradients["layer"+str(self.num_layers-1)] = (y - self.xs["layer"+str(self.num_layers-1)]) * \
                                                         sigmoid_prime(self.x_hats["layer"+str(self.num_layers-1)])
        for i in reversed(range(1, self.num_layers-1)):
            self.gradients["layer"+str(i)] = sigmoid_prime(self.x_hats["layer"+str(i)]) * \
                                        self.parameters["w"+str(i+1)].T.dot(self.gradients["layer"+str(i+1)])

    def update(self, alpha, lr):
        for i in range(1, self.num_layers):
            self.moments["w"+str(i)] = alpha * self.moments["w"+str(i)] + \
                                       lr * np.outer(self.gradients["layer"+str(i)], self.xs["layer"+str(i-1)])
            self.parameters["w"+str(i)] += self.moments["w"+str(i)]
            self.moments["b"+str(i)] = alpha * self.moments["b"+str(i)] + lr * self.gradients["layer"+str(i)]
            self.parameters["b"+str(i)] += self.moments["b"+str(i)]

class MLQP:
    def __init__(self, layers_dim):
        self.num_layers = len(layers_dim)
        self.parameters = {}
        self.gradients = {}
        self.moments = {}
        self.x_hats = {}
        self.xs = {}
        for i in range(1, self.num_layers):
            self.parameters["u"+str(i)] = np.random.normal(size=(layers_dim[i], layers_dim[i-1]))
            self.parameters["v"+str(i)] = np.random.normal(size=(layers_dim[i], layers_dim[i-1]))
            self.parameters["b"+str(i)] = np.random.normal(size=(layers_dim[i], 1))
            self.moments["u"+str(i)] = np.zeros((layers_dim[i], layers_dim[i-1]))
            self.moments["v"+str(i)] = np.zeros((layers_dim[i], layers_dim[i-1]))
            self.moments["b"+str(i)] = np.zeros((layers_dim[i], 1))

    def forward(self, x):
        x = x[:, np.newaxis]
        self.x_hats["layer0"] = x # non-sense
        self.xs["layer0"] = x
        for i in range(1, self.num_layers):
            x_hat = self.parameters["u"+str(i)].dot(x*x) + self.parameters["v"+str(i)].dot(x) \
                    + self.parameters["b"+str(i)]
            x = sigmoid(x_hat)
            self.x_hats["layer"+str(i)] = x_hat
            self.xs["layer"+str(i)] = x
        return x

    def backward(self, y):
        self.gradients["layer"+str(self.num_layers-1)] = (y-self.xs["layer"+str(self.num_layers-1)]) * \
                                                         sigmoid_prime(self.x_hats["layer"+str(self.num_layers-1)])
        for i in reversed(range(1, self.num_layers-1)):
            self.gradients["layer"+str(i)] = sigmoid_prime(self.x_hats["layer"+str(i)]) * \
                                             ((2 * self.xs["layer"+str(i)]) * (self.parameters["u"+str(i+1)].T.dot(self.gradients["layer"+str(i+1)])) + \
                                              self.parameters["v"+str(i+1)].T.dot(self.gradients["layer"+str(i+1)]))

    def update(self, alpha1, alpha2, lr1, lr2):
        for i in range(1, self.num_layers):
            self.moments["u"] = alpha2 * self.moments["u"+str(i)] + \
                                lr2 * np.outer(self.gradients["layer"+str(i)], self.xs["layer"+str(i-1)]*self.xs["layer"+str(i-1)])
            self.parameters["u"+str(i)] += self.moments["u"]
            self.moments["v"] = alpha1 * self.moments["v"+str(i)] + \
                                lr1 * np.outer(self.gradients["layer"+str(i)], self.xs["layer"+str(i-1)])
            self.parameters["v"+str(i)] += self.moments["v"]
            self.moments["b"+str(i)] = alpha1 * self.moments["b"+str(i)] + (1-alpha1) * lr1 * self.gradients["layer"+str(i)]
            self.parameters["b"+str(i)] += self.moments["b"+str(i)]



