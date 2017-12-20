from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers, losses, callbacks

from utils import *

# following https://blog.keras.io/building-autoencoders-in-keras.html
class Autoencoder():
    def __init__(self, input_dim = 784, encoding_dim = 36, weight_decay = True, sparse = False, contractive = False, denoising = False, robust = False, activation = "tanh"):
        self.weight_decay = weight_decay
        self.sparse = sparse
        self.contractive = contractive
        self.denoising = denoising
        self.robust = robust

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.activation = activation

        attrs = []
        if self.sparse: attrs.append("sparse")
        if self.contractive: attrs.append("contractive")
        if self.denoising: attrs.append("denoising")
        if self.robust: attrs.append("robust")
        if not attrs: attrs.append("basic")
        if self.weight_decay: attrs.append("wd")

        self.name = "{}-{}-{}".format(
            "-".join(attrs),
            encoding_dim,
            activation
        )

        self.build()

    def build(self):
        # "encoded" is the encoded representation of the input
        activity_reg = KLregularizer(-0.7, 0.2) if self.sparse else regularizers.l1(0.)
        kernel_reg = regularizers.l2(0.02) if self.weight_decay else regularizers.l1(0.)
        
        # this is our input placeholder
        input_img = Input(shape=(self.input_dim,))
        
        encoded = Dense(self.encoding_dim, activation=self.activation
                        , activity_regularizer = activity_reg
                        , kernel_regularizer = kernel_reg
                        , name = "encoded")(input_img)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(self.input_dim, activation=('linear' if self.activation == "linear" else 'sigmoid')
                        , name = "decoded")(encoded)

        # this model maps an input to its reconstruction
        self.model = Model(input_img, decoded)

        # this model maps an input to its encoded representation
        self.encoder = Model(input_img, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.model.layers[-1]
        # create the decoder model
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
