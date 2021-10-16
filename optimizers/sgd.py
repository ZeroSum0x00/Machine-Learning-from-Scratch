

class SGD:
    def __init__(self, learning_rate=0.4, momentum=0.0, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = 0.0

    def caculator(self, weights, bias, dw, db):
        for i, (dw_epoch, db_epoch) in enumerate(zip(dw, db)):
            if self.momentum > 0:
                self.velocity = self.momentum * self.velocity - self.learning_rate * dw_epoch
                if self.nesterov:
                    weights[i] += self.momentum * self.velocity - self.learning_rate * dw_epoch
                else:
                    weights[i] += self.velocity
            else:
                weights[i] -= self.learning_rate * dw_epoch

            bias[i] -= self.learning_rate * db_epoch

        return weights, bias
