from module import Linear, ReLU, Sigmoid, SoftMax, Sequentiel, TanH
from utils import confusion_matrix, one_hot
from loss import CrossEntropy
from optimizer import SGD
from trainer import Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

import argparse

np.random.seed(1)

with open("../data/MNIST_data.npy", "rb") as f:
    X = np.load(f, allow_pickle=True)
with open("../data/MNIST_label.npy", "rb") as f:
    Y = np.load(f, allow_pickle=True)

X = X / 255
Y = Y.astype("int")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2)

y_train = one_hot(y_train)
y_val = one_hot(y_val)

def main(activation="ReLU", early_stop=True):
    if activation ==  "ReLU":
        m = Sequentiel(Linear(784, 392), ReLU(), Linear(392, 196), ReLU(), Linear(196, 98), ReLU(), Linear(98, 49), ReLU(), Linear(49, 10))
    elif activation == "Sigmoid":
        m = Sequentiel(Linear(784, 392), Sigmoid(), Linear(392, 196), Sigmoid(), Linear(196, 98), Sigmoid(), Linear(98, 49), Sigmoid(), Linear(49, 10))
    else:
        m = Sequentiel(Linear(784, 392), TanH(), Linear(392, 196), TanH(), Linear(196, 98), TanH(), Linear(98, 49), TanH(), Linear(49, 10))

    loss = CrossEntropy()
    op = SGD(m, loss, lr=.1)

    trainer = Trainer(op, loss, 100, x_train, y_train, x_val, y_val, early_stop)
    train_loss, val_loss = trainer.run()
    trainer.save("mnist_multi_class" + activation)

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label="Train loss")
    plt.plot(range(len(val_loss)), val_loss, label="Validation loss")
    plt.title(f"Train - validation loss on MNIST using {activation}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if early_stop:
        plt.savefig(f"./plots/mnist_loss_{activation}")
    else:
        plt.savefig(f"./plots/mnist_loss_{activation}_no_early_stop")

    m.add_layer(SoftMax())
    yhat = np.argmax(m.forward(x_test), axis=1)

    cm = confusion_matrix(y_test, yhat)

    fig = plt.figure()

    plt.title(f"Confusion matrix - {activation} on MNIST\nTest accuracy: {accuracy_score(y_test, yhat):.2f}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True label")

    im = plt.imshow(cm)

    fig.colorbar(im)

    if early_stop:
        plt.savefig(f"./plots/mnist_conf_mat_{activation}")
    else:
        plt.savefig(f"./plots/mnist_conf_mat_{activation}_no_early_stop")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test arguments")

    parser.add_argument("--activation", type=str, default="ReLU",
                        choices=["ReLU", "Sigmoid", "TanH"])

    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--no-early_stop", dest="early_stop", action="store_false")
    parser.set_defaults(early_stop=True)

    args = parser.parse_args()
    activation, early_stop = args.activation, args.early_stop

    if activation == "ReLU":
        main(early_stop=early_stop)
    elif activation == "Sigmoid":
        main("Sigmoid", early_stop=early_stop)
    else:
        main("TanH", early_stop=early_stop)

