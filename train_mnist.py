import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from scratch_grad import Variable

if __name__ == '__main__':
    # Random seed
    np.random.seed(42)
    random.seed(42)

    # Training parameters
    num_epochs = 30
    learning_rate = 1e-3

    # Network
    # TODO creer votre reseau
    l1_unit = 784
    l2_unit = 1000
    w_1_data = np.random.randn(784, l1_unit).astype(np.float32)
    w_2_data = np.random.randn(l1_unit, l2_unit).astype(np.float32)
    w_3_data = np.random.randn(l2_unit, 10).astype(np.float32)
    b_1_data = np.random.randn(1, l1_unit).astype(np.float32)
    b_2_data = np.random.randn(1, l2_unit).astype(np.float32)
    b_3_data = np.random.randn(1, 10).astype(np.float32)
    sg_w_1 = Variable(w_1_data, name='w1')
    sg_b_1 = Variable(b_1_data, name='b1')
    sg_w_2 = Variable(w_2_data, name='w2')
    sg_b_2 = Variable(b_2_data, name='b2')
    sg_w_3 = Variable(w_3_data, name='w3')
    sg_b_3 = Variable(b_3_data, name='b3')

    # Dataset
    mnist = MNIST(root='data/', train=True, download=True)
    mnist.transform = ToTensor()

    mnist_test = MNIST(root='data/', train=False, download=True)
    mnist_test.transform = ToTensor()

    # Only take a small subset of MNIST
    mnist = Subset(mnist, range(len(mnist) // 16))
    mnist_test = Subset(mnist_test, range(32))

    # Dataloaders
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True)
    val_loader = DataLoader(mnist_test, batch_size=1)

    # Logging
    train_loss_by_epoch = []
    train_acc_by_epoch = []
    val_loss_by_epoch = []
    val_acc_by_epoch = []

    epochs = list(range(num_epochs))
    for epoch in epochs:
        train_predictions = []
        train_losses = []
        # Training loop
        for x, y in tqdm.tqdm(train_loader):
            # Put the data into Variables
            x = x.numpy().reshape((1, 784))
            x = Variable(np.array(x), name='x')
            y = y.numpy().reshape((1, 1))
            y = Variable(np.array(y), name='y')

            # Pass the data through the network
            # TODO passer les valeurs dans le reseau
            z_1 = (x @ sg_w_1 + sg_b_1).relu()
            z_2 = (z_1 @ sg_w_2 + sg_b_2).relu()
            z_3 = z_2 @ sg_w_3 + sg_b_3

            # Compute the loss
            # TODO calculer la fonction de perte
            loss = z_3.nll(y)

            # Apply backprop
            # TODO appliquer la backprop sur la perte
            loss.backward()

            # Update the weights
            # TODO mettre-a-jour les poids avec un learning rate de `learning_rate`
            with torch.no_grad():
                sg_w_1.data -= learning_rate * sg_w_1.grad
                sg_w_2.data -= learning_rate * sg_w_2.grad
                sg_w_3.data -= learning_rate * sg_w_3.grad
                sg_b_1.data -= learning_rate * sg_b_1.grad
                sg_b_2.data -= learning_rate * sg_b_2.grad
                sg_b_3.data -= learning_rate * sg_b_3.grad

            # Reset gradients
            # TODO mettre les gradients a zero avec `variable.zero_grad()`
            sg_w_1.zero_grad()
            sg_w_2.zero_grad()
            sg_w_3.zero_grad()
            sg_b_1.zero_grad()
            sg_b_2.zero_grad()
            sg_b_3.zero_grad()

            loss.show()

            # Logging
            train_losses.append(loss.data)
            train_predictions.append(np.argmax(z_3.data, axis=1) == y.data)

        # Validation loop
        val_results = []
        val_losses = []
        for x, y in val_loader:
            # Put the data into Variables
            x = x.numpy().reshape((1, 784))
            x = Variable(np.array(x), name='x')
            y = y.numpy().reshape((1, 1))
            y = Variable(np.array(y), name='y')

            # Pass the data through the network
            # TODO passer les valeurs dans le reseau
            z_1 = (x @ sg_w_1 + sg_b_1).relu()
            z_2 = (z_1 @ sg_w_2 + sg_b_2).relu()
            z_3 = z_2 @ sg_w_3 + sg_b_3

            # Compute the loss
            # TODO calculer la fonction de perte
            loss = z_3.nll(y)

            # Logging
            val_losses.append(loss.data)
            val_results.append(np.argmax(z_3.data, axis=1) == y.data)

        # Compute epoch statistics
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_predictions)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_results)

        # Show progress
        print(f'Epoch {epoch}')
        print(f'\tTrain:\t\tLoss {train_loss},\tAcc {train_acc}')
        print(f'\tValidation:\tLoss {val_loss},\tAcc {val_acc}')

        # Logging
        train_loss_by_epoch.append(train_loss)
        train_acc_by_epoch.append(train_acc)
        val_loss_by_epoch.append(val_loss)
        val_acc_by_epoch.append(val_acc)

    # Draw the accuracy-loss plot
    _, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_ylabel('Accuracy')
    axes[0].plot(epochs, train_acc_by_epoch, label='Train')
    axes[0].plot(epochs, val_acc_by_epoch, label='Validation')
    axes[0].legend()

    axes[1].set_ylabel('Loss')
    axes[1].plot(epochs, train_loss_by_epoch, label='Train')
    axes[1].plot(epochs, val_loss_by_epoch, label='Validation')

    axes[1].set_xlabel('Epochs')
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
