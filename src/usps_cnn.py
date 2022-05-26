from module import Sequentiel, Linear, Flatten, Conv2D, MaxPool, ReLU, SoftMax
from loss import CrossEntropy
from trainer import Trainer
from optimizer import SGD

from utils import one_hot, load_usps

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

X, Y = load_usps("../data/USPS_train.txt")
X = X.reshape(X.shape[0], 16, 16, 1)
X_test, Y_test = load_usps("../data/USPS_test.txt")
X_test = X_test.reshape(X_test.shape[0], 16, 16, 1)

x_train, x_val, y_train, y_val = train_test_split(X, one_hot(Y), test_size=.2)

def main():
    m = Sequentiel(
        Conv2D(kernel_size=3, n_chanel=1, n_kernel=32, stride=1, pad=2),
        ReLU(),
        MaxPool(kernel_size=3, stride=2),
        Flatten(),
        Linear(2048, 100),
        ReLU(),
        Linear(100, 10)
    )

    loss = CrossEntropy()

    op = SGD(m, loss, lr=.01)

    trainer = Trainer(op, loss, 20, x_train, y_train, x_val, y_val, early_stop=False)

    train_loss, val_loss = trainer.run()
    
    trainer.save("usps_cnn")

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label="Train loss")
    plt.plot(range(len(val_loss)), val_loss, label="Validation loss")
    plt.title(f"Train - validation loss on USPS using simple CNN")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"../plots/usps-cnn-train_val_loss")

if __name__ == "__main__":
    main()

