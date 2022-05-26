from module import Sequentiel, Linear, Flatten, Conv2D, MaxPool, ReLU, SoftMax
from loss import CrossEntropy
from trainer import Trainer
from optimizer import SGD

from utils import one_hot

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784")

X, Y = mnist["data"].to_numpy(), mnist["target"].to_numpy()

X = X / 255.0
X = X.reshape(X.shape[0], 28, 28, 1)
Y = Y.astype("int")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, one_hot(y_train), test_size=.2)

def main():
    m = Sequentiel(
        Conv2D(kernel_size=5, n_chanel=1, n_kernel=6, stride=1, pad=2),
        ReLU(),
        MaxPool(kernel_size=2, stride=2),
        Conv2D(kernel_size=5, n_chanel=6, n_kernel=16, stride=1, pad=0),
        ReLU(),
        MaxPool(kernel_size=2, stride=2),
        Flatten(),
        Linear(400, 120),
        ReLU(),
        Linear(120, 10)
    )

    loss = CrossEntropy()

    op = SGD(m, loss, lr=.01, batch_size=128)

    trainer = Trainer(op, loss, 50, x_train, y_train, x_val, y_val, early_stop=True)

    train_loss, val_loss = trainer.run()
    
    trainer.save("lenet-5")

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label="Train loss")
    plt.plot(range(len(val_loss)), val_loss, label="Validation loss")
    plt.title(f"Train - validation loss of LeNet-5 on MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"../plots/lenet5-train_val_loss")

if __name__ == "__main__":
    main()

