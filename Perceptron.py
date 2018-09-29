import numpy as np


class Perceptron:

    def __init__(self, eta=0.1, epochs=50, random_state=1, shape=[None, 2]):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

        rgen = np.random.RandomState(self.random_state)
        self.weight = rgen.normal(loc=0.0, scale=0.01, size=shape[1])
        self.baias = rgen.normal(loc=0.0, scale=0.01, size=shape[1])[0]
        self.errors = []

    def fit(self, x, y):
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(x, y):
                delta_w = self.eta * (target - self.predict(xi))
                self.weight += delta_w * xi
                self.baias += delta_w
                errors += abs(delta_w)

            self.errors.append(errors)

    def input(self, x):
        return np.dot(x, self.weight) + self.baias

    def predict(self, x):
        return np.where(self.input(x) >= 0.0, 1, -1)

    def test(self, x, y):
        num_error = 0
        for i in range(x.__len__()):
            if self.predict(x[i]) != y[i]:
                num_error += 1

            out = self.predict(x[i])
            print(out, y[i])