from module import Linear, ReLU, Sigmoid, Sequentiel, TanH
from loss import BinaryCrossEntropyLoss
from optimizer import SGD
from trainer import Trainer

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt
import numpy as np

import argparse

np.random.seed(1)

mnist = fetch_openml("mnist_784")
X, Y = mnist["data"].to_numpy(), mnist["target"].to_numpy()

X = X / 255
Y = Y.astype("int")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2)

def main(activation="TanH"):
    if activation == "TanH":
        m = Sequentiel(Linear(784, 392), TanH(), Linear(392, 98), TanH(), Linear(98, 49), TanH(), Linear(49, 10), TanH(),
                    Linear(10, 49), TanH(), Linear(49, 98), TanH(), Linear(98, 392), TanH(), Linear(392, 784), Sigmoid())
    elif activation == "ReLU":
        m = Sequentiel(Linear(784, 392), ReLU(), Linear(392, 98), ReLU(), Linear(98, 49), ReLU(), Linear(49, 10), ReLU(),
                    Linear(10, 49), ReLU(), Linear(49, 98), ReLU(), Linear(98, 392), ReLU(), Linear(392, 784), Sigmoid())
    else:
        m = Sequentiel(Linear(784, 392), Sigmoid(), Linear(392, 98), Sigmoid(), Linear(98, 49), Sigmoid(), Linear(49, 10), Sigmoid(),
                    Linear(10, 49), Sigmoid(), Linear(49, 98), Sigmoid(), Linear(98, 392), Sigmoid(), Linear(392, 784), Sigmoid())

    loss = BinaryCrossEntropyLoss()
    op = SGD(m, loss, lr=.1)

    trainer = Trainer(op, loss, 60, x_train, x_train, x_val, x_val, early_stop=False)
    train_loss, val_loss = trainer.run()
    trainer.save(f"auto_encoder_{activation}")

    plt.figure()

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label="Train loss")
    plt.plot(range(len(val_loss)), val_loss, label="Validation loss")
    plt.title(f"Train - validation loss of Auto-Encoder on MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"../plots/auto_encoder_loss_{activation}")
    plt.legend()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="activation")
    parser.add_argument("--activation", type=str, default="ReLU",
                        choices=["ReLU", "Sigmoid", "TanH"])

    activation = parser.parse_args()

    main(activation)