
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, y_test) = mnist.load_data()
# normalize
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# flatten
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

def find_first_digit(d):
    return next(i for i in range(len(y_test)) if y_test[i] == d)

indices = [find_first_digit(d) for d in range(10)]

from sklearn.decomposition import PCA

pca = PCA(36)

encoded_imgs = pca.fit(x_train).transform(x_test)
decoded_imgs = pca.inverse_transform(encoded_imgs)

n = len(indices)  # how many digits we will display
fig = plt.figure(figsize=(20, 4))

rows = 3
side = 6

for i in range(n):

    ax = plt.subplot(rows, n, i + 1)
    plt.imshow(x_test[indices[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoding
    ax = plt.subplot(rows, n, i + 1 + n)
    plt.imshow(((encoded_imgs[indices[i]] + 1) / 2).reshape(side, side))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(rows, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[indices[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fig.savefig("/home/fdavidcl/Documentos/research/publications/2017/ReviewAutoencoders/examples/pca-36.pdf", pad_inches = 0)

