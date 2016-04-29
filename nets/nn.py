import numpy as np
from sklearn.metrics import log_loss


class NeuralNet(object):

    def __init__(self, activation, activation_prime, no_input=2,
                 hidden_layers=[5], no_output=2):
        self._activation = activation
        self._activation_prime = activation_prime

        self._layers = []
        self._biases = []
        for i, h in enumerate(hidden_layers):
            if i == 0:
                layer = 2 * np.random.rand(no_input, h) - 1
            else:
                layer = 2 * np.random.rand(hidden_layers[i-1], h) - 1

            self._layers.append(layer)
            self._biases.append(np.zeros(h))

        self._layers.append(
            2 * np.random.rand(hidden_layers[-1], no_output) - 1
        )
        self._biases.append(np.zeros(no_output))

    def train(self, X, y, epoch=500, eps=0.01, reg_lambda=0.01,
              print_loss=False, plt=None):
        if plt is not None:
            loss = []
        y = self._onehot(y)
        l = len(self._layers)
        for i in range(0, epoch):
            # Forward Propagation
            a = {}
            z = {}
            for j, w in enumerate(self._layers):
                if j == 0:
                    tmp = X.dot(w) + self._biases[j]
                else:
                    tmp = a[j-1].dot(w) + self._biases[j]

                z[j] = tmp

                if j == l - 1:
                    a[j] = self._softmax(tmp)
                else:
                    a[j] = self._activation(tmp)

            if plt is not None:
                loss.append(log_loss(y, a[l-1]))

            if print_loss and (i) % 100 == 0:
                ll = log_loss(y, a[l-1])
                print("Epoch {}. Loss {}".format(i, ll))

            d = {}
            dW = {}
            db = {}
            # start at last layer and go backwards
            for j in range(l-1, -1, -1):
                if j == l-1:
                    d[j] = np.subtract(a[j], y)
                else:
                    d[j] = np.multiply(d[j+1].dot(self._layers[j+1].T),
                                       self._activation_prime(z[j]))

            for j in range(l):
                if j == 0:
                    dW[j] = (X.T).dot(d[j])
                else:
                    dW[j] = (a[j-1].T).dot(d[j])

                db[j] = np.sum(d[j], axis=0)

            for j in range(l):
                dW[j] += reg_lambda * self._layers[j]

            for j in range(l):
                self._layers[j] += -eps * dW[j]
                self._biases[j] += -eps * np.squeeze(np.asarray(db[j]))

        if plt is not None:
            plt.plot(range(epoch), loss)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

    def predict(self, x):
        l = len(self._layers)
        a = {}
        z = {}
        for j, w in enumerate(self._layers):
            if j == 0:
                tmp = x.dot(w) + self._biases[j]
            else:
                tmp = a[j-1].dot(w) + self._biases[j]

            z[j] = tmp

            if j == l - 1:
                a[j] = self._softmax(tmp)
            else:
                a[j] = self._activation(tmp)
        return np.argmax(a[l-1], axis=1)

    def _softmax(self, z):
        e = np.exp(np.array(z))
        return e / np.sum(e, axis=1, keepdims=True)

    def _onehot(self, y):
        oh = np.zeros((np.max(y)+1, np.max(y)+1))
        np.fill_diagonal(oh, 1)
        y = [list(oh[h]) for h in y]
        return np.matrix(y)
