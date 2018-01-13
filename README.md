# Autoencoder review resources

Additional resources for the work "A practical tutorial on autoencoders for nonlinear feature fusion: Taxonomy, models, software and guidelines" (DOI: [10.1016/j.inffus.2017.12.007](https://doi.org/10.1016/j.inffus.2017.12.007), arXiv: [1801.01586](https://arxiv.org/abs/1801.01586)).

## Contents

This repository contains a series of examples on different types of autoencoders built on top of [Keras](https://github.com/keras-team/keras).

- `autoencoder.py` defines an `Autoencoder` class which describes the model that will be learned out of the data.
- `utils.py` defines additional functionality needed the sparse, contractive, denoising and robust autoencoders.
- `mnist.py` includes the training process for autoencoders with MNIST data.
- `cancer.py` trains autoencoder with the Wisconsin Breast Cancer Diagnosis data set (file `wdbc.arff` is needed for the script to run).
- `pca.py` outputs the result of Principal Component Analysis with MNIST for comparison purposes.

## Autoencoders

Autoencoders are symmetrical neural networks trained to reconstruct their inputs onto their outputs, with some restrictions that force them to find meaningful codifications of data. These examples cover several types of autoencoders:

- Basic, undercomplete autoencoder
- Basic autoencoder with weight decay
- [Sparse autoencoder*](https://pdfs.semanticscholar.org/eb2f/e260af00818907fe82024203d8a5a1386777.pdf)
- [Contractive autoencoder*](https://dl.acm.org/citation.cfm?id=3104587)
- [Denoising autoencoder](https://dl.acm.org/citation.cfm?id=1390294)
- [Robust autoencoder](https://ieeexplore.ieee.org/abstract/document/6854900/)

The features of these can be combined (for example, one can build a sparse denoising autoencoder with weight decay). Examples of this can be found inside file `mnist.py`.

(*) Only for sigmoid and tanh activation functions in the encoding layer.

## Credits and licensing

Other resources used:

- https://blog.keras.io/building-autoencoders-in-keras.html
- https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
- https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
- [Keras docs](https://keras.io/)
- [Scikit-learn docs](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Tensorflow docs](https://www.tensorflow.org/api_docs/python/tf/cond)

The code in this repository is licensed under [MPL v2.0](https://github.com/fdavidcl/ae-review-resources/blob/master/LICENSE), which allows you to redistribute and use it in your own work.