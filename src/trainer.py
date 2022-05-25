import numpy as np
import os
import pickle

class Trainer(object):
    def __init__(self, optimizer, loss, max_epochs, x_train, y_train, x_val, y_val, eval=2, early_stop=True):
        self.op = optimizer
        self.loss = loss
        self.max_epochs = max_epochs
        self.x_train, self.y_train = x_train, y_train
        try:
            self.train_size = y_train.shape[0] * y_train.shape[1]
        except:
            self.train_size = y_train.shape[0]
        self.x_val, self.y_val = x_val, y_val
        try:
            self.val_size = y_val.shape[0] * y_val.shape[1]
        except:
            self.val_size = y_val.shape[0]
        self.train_loss = []
        self.val_loss = []
        self.min_val_loss = float("inf")
        self.last_min_val_loss = float("inf")
        self.eval = 2
        self.early_stop = early_stop

    def run(self):
        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch}...")
            for batch in self.op.iterate_mini_batch(self.x_train, self.y_train):
                batch_x, batch_y = batch
                self.op.step(batch_x, batch_y, epoch+1)
                self.op.zero_grad()

            self.train_loss.append(np.sum(self.loss.forward(self.y_train, self.op.net.forward(self.x_train))) / self.train_size)
            self.val_loss.append(np.sum(self.loss.forward(self.y_val, self.op.net.forward(self.x_val))) / self.val_size)

            if self.early_stop:
                if self.val_loss[epoch] < self.min_val_loss:
                    self.min_val_loss = self.val_loss[epoch]
                
                if epoch == 0:
                    self.last_min_val_loss = self.min_val_loss

                if (epoch + 1) % self.eval == 0:
                    # print(self.min_val_loss, self.last_min_val_loss)
                    if abs(self.min_val_loss - self.last_min_val_loss) < 1e-5 or self.min_val_loss - self.last_min_val_loss > 1e-5:
                        print(f"Early stopping triggered at epoch {epoch}")
                        return self.train_loss, self.val_loss
                    else:
                        self.last_min_val_loss = self.min_val_loss

        return self.train_loss, self.val_loss

    def save(self, filename, dir="../archives/"):
        filename = dir + filename

        if not os.path.exists(filename):
            os.makedirs(filename)

        with open(filename + "/train_loss", "wb") as f1:
            np.save(f1, self.train_loss)

        with open(filename + "/val_loss", "wb") as f2:
            np.save(f2, self.val_loss)

        with open(filename + "/model", "wb") as f3:
            pickle.dump(self.op.net, f3)
