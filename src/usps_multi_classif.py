from module import Linear, ReLU, Sigmoid, SoftMax, Sequentiel, TanH
from utils import load_usps, confusion_matrix, one_hot
from loss import CrossEntropy
from optimizer import SGD
from trainer import Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

import argparse

np.random.seed(1)

X, Y = load_usps("../data/USPS_train.txt")
Y_onehot = one_hot(Y)

X_test, Y_test = load_usps("../data/USPS_test.txt")

x_train, x_val, y_train, y_val = train_test_split(X, Y_onehot, test_size=.2)

def main(activation="ReLU", early_stop=True):
    if activation ==  "ReLU":
        m = Sequentiel(Linear(256, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 32), ReLU(), Linear(32, 10))
    elif activation == "Sigmoid":
        m = Sequentiel(Linear(256, 128), Sigmoid(), Linear(128, 64), Sigmoid(), Linear(64, 32), Sigmoid(), Linear(32, 10))
    else:
        m = Sequentiel(Linear(256, 128), TanH(), Linear(128, 64), TanH(), Linear(64, 32), TanH(), Linear(32, 10))

    loss = CrossEntropy()
    op = SGD(m, loss, lr=.1)

    trainer = Trainer(op, loss, 100, x_train, y_train, x_val, y_val, early_stop)
    train_loss, val_loss = trainer.run()
    trainer.save("usps_multi_class" + activation)

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label="Train loss")
    plt.plot(range(len(val_loss)), val_loss, label="Validation loss")
    plt.title(f"Train - validation loss on USPS using {activation}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if early_stop:
        plt.savefig(f"./plots/usps_loss_{activation}")
    else:
        plt.savefig(f"./plots/usps_loss_{activation}_no_early_stop")

    m.add_layer(SoftMax())
    yhat = np.argmax(m.forward(X_test), axis=1)

    cm = confusion_matrix(Y_test, yhat)

    fig = plt.figure()

    plt.title(f"Confusion matrix - {activation} on USPS\nTest accuracy: {accuracy_score(Y_test, yhat):.2f}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True label")

    im = plt.imshow(cm)

    fig.colorbar(im)

    if early_stop:
        plt.savefig(f"./plots/usps_conf_mat_{activation}")
    else:
        plt.savefig(f"./plots/usps_conf_mat_{activation}_no_early_stop")

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

