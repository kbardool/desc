import math
import os

import torch

from ._nn import (
    INITIALIZERS,
    SGD,
    Dense,
    Dropout,
    EarlyStopping,
    Sequential,
    as_float32_array,
    fit,
    mse,
    predict,
    set_seed,
)


class SAE(object):
    """
    Stacked autoencoders. It can be trained in layer-wise manner followed by end-to-end fine-tuning.
    For a 5-layer (including input layer) example:
        Autoendoers model: Input -> encoder_0->act -> encoder_1 -> decoder_1->act -> decoder_0;
        stack_0 model: Input->dropout -> encoder_0->act->dropout -> decoder_0;
        stack_1 model: encoder_0->act->dropout -> encoder_1->dropout -> decoder_1->act;
    The stacks share their encoder_i/decoder_i layers with the autoencoders model, so layer-wise
    pretrained weights are directly used for the end-to-end fine-tuning.

    Usage:
        from desc.models.SAE import SAE
        sae = SAE(dims=[784, 500, 10])  # define a SAE with 5 layers
        sae.fit(x, epochs=100)
        features = sae.extract_feature(x)

    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
              The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation (default='relu'), not applied to Input, Hidden and Output layers.
        drop_rate: drop ratio of Dropout for constructing denoising autoencoder 'stack_i' during layer-wise pretraining
        batch_size: `int`, optional. Default:`32`, the batch size for autoencoder model.
        random_seed, `int`,optional. Default,`201809`. the random seed for random.seed, numpy.random.seed and torch.manual_seed
        actincenter: the activation function in last layer for encoder and last layer for encoder (avoiding the representation values and reconstruct outputs are all non-negative)
        init: `str`,optional. Default: `glorot_uniform`. Initialization method used to initialize weights.
        use_earlyStop: optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.
        save_dir:'str',optional. Default,'result_tmp',some result will be saved in this directory.
        device: `str` or `torch.device`, optional. Default, `None` (CPU). The device used for training.
    """
    def __init__(self, dims, act='relu',
            drop_rate=0.2,
            batch_size=32,
            random_seed=201809,
            actincenter="tanh",
            init="glorot_uniform",
            use_earlyStop=True,
            save_dir='result_tmp',
            device=None): #act relu
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.n_layers = 2*self.n_stacks  # exclude input layer
        self.activation = act
        self.actincenter=actincenter #linear
        self.drop_rate = drop_rate
        self.init=self.choose_init(init)
        self.batch_size = batch_size
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        #set random seed
        set_seed(random_seed)
        os.makedirs(save_dir, exist_ok=True)
        #
        self.random_seed=random_seed
        self.use_earlyStop=use_earlyStop
        self.layers = self.make_layers()
        self.stacks = [self.make_stack(i,random_seed=self.random_seed+2*i) for i in range(self.n_stacks)]
        self.autoencoders ,self.encoder= self.make_autoencoders()
        self.autoencoders.to(self.device)

    def choose_init(self,init="glorot_uniform",seed=1):
        if init not in INITIALIZERS:
            raise ValueError('Invalid `init` argument: expected one of %s but got %r' % (INITIALIZERS, init))
        return init

    def make_layers(self):
        """ The encoder_i/decoder_i layers shared by the stacks and the autoencoders model.
        """
        layers = {}
        for i in range(self.n_stacks):
            # hidden layer and output layer use actincenter (tanh, or linear), others use act
            hidden_act = self.actincenter if i == self.n_stacks - 1 else self.activation
            output_act = self.actincenter if i == 0 else self.activation
            layers['encoder_%d' % i] = Dense(self.dims[i], self.dims[i + 1], activation=hidden_act, kernel_initializer=self.init, name='encoder_%d' % i)
            layers['decoder_%d' % i] = Dense(self.dims[i + 1], self.dims[i], activation=output_act, kernel_initializer=self.init, name='decoder_%d' % i)
        return layers

    def make_autoencoders(self):
        """ Fully connected autoencoders model, symmetric.
        """
        encoder_layers = [self.layers['encoder_%d' % i] for i in range(self.n_stacks)]  # features are extracted from the last one
        decoder_layers = [self.layers['decoder_%d' % i] for i in range(self.n_stacks - 1, -1, -1)]
        return Sequential(encoder_layers + decoder_layers, name="AE"), Sequential(encoder_layers, name="encoder")

    def make_stack(self, ith,random_seed=0):
        """
        Make the ith denoising autoencoder for layer-wise pretraining. It has single hidden layer. The input data is
        corrupted by Dropout(drop_rate)

        Arguments:
            ith: int, in [0, self.n_stacks)
            random_seed: unused, kept for backward compatibility (dropout uses the torch random generator)
        """
        return Sequential([
            Dropout(self.drop_rate, name='dropout_input_%d' % ith),
            self.layers['encoder_%d' % ith],
            Dropout(self.drop_rate, name='dropout_hidden_%d' % ith),
            self.layers['decoder_%d' % ith],
        ], name='stack_%d' % ith)

    def pretrain_stacks(self, x, epochs=200,decaying_step=3):
        """
        Layer-wise pretraining. Each stack is trained for 'epochs' epochs using SGD with learning rate decaying 10
        times every 'epochs/3' epochs.

        Arguments:
            x: input data, shape=(n_samples, n_dims)
            epochs: epochs for each stack
            decayiing_step: learning rate multiplies 0.1 every 'epochs/decaying_step' epochs
        """
        features = as_float32_array(x)
        for i in range(self.n_stacks):
            print( 'Pretraining the %dth layer...' % (i+1))
            for j in range(int(decaying_step)):  # learning rate multiplies 0.1 every 'epochs/decaying_step' epochs
                print ('learning rate =', pow(10, -1-j))
                optimizer = SGD(pow(10, -1-j), momentum=0.9).build(self.stacks[i].parameters())
                early_stopping = EarlyStopping(min_delta=1e-4, patience=10) if self.use_earlyStop is True else None
                fit(self.stacks[i], features, features, optimizer, mse, batch_size=self.batch_size, epochs=math.ceil(epochs/decaying_step), early_stopping=early_stopping)
            print ('The %dth layer has been pretrained.' % (i+1))

            # update features to the inputs of the next layer
            features = predict(self.layers['encoder_%d' % i], features)

    def pretrain_autoencoders(self, x, epochs=300):
        """
        Fine tune autoendoers end-to-end after layer-wise pretraining using 'pretrain_stacks()'
        Use SGD with learning rate = 0.1, decayed 10 times every 50 epochs

        Arguments:
        x: input data, shape=(n_samples, n_dims)
        epochs: training epochs
        """
        print ('Fine-tuning autoencoder end-to-end')
        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -j)
            print ('learning rate =', lr)
            optimizer = SGD(lr, momentum=0.9).build(self.autoencoders.parameters())
            fit(self.autoencoders, x, x, optimizer, mse, batch_size=self.batch_size, epochs=50, early_stopping=EarlyStopping(min_delta=1e-4, patience=10))

    def fit(self, x, epochs=300,decaying_step=3): # use stacked autoencoder pretrain and fine tuning
        self.pretrain_stacks(x, epochs=int(epochs/2),decaying_step=decaying_step)
        self.pretrain_autoencoders(x, epochs=epochs)

    def fit2(self,x,epochs=300): #no stack directly tran
        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -j)
            print ('learning rate =', lr)
            optimizer = SGD(lr, momentum=0.9).build(self.autoencoders.parameters())
            early_stopping = EarlyStopping(min_delta=1e-4, patience=10) if self.use_earlyStop else None
            fit(self.autoencoders, x, x, optimizer, mse, batch_size=self.batch_size, epochs=50, early_stopping=early_stopping)

    def extract_feature(self, x):
        """
        Extract features from the middle layer of autoencoders(representation).

        Arguments:
        x: data
        """
        return self.encoder.predict(x)


if __name__ == "__main__":
    """
    An example for how to use SAE model on the scikit-learn digits dataset. Run `python -m desc.models.SAE` in terminal
    """
    from sklearn.datasets import load_digits
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi

    x, y = load_digits(return_X_y=True)
    x = x / 16.0
    print ('digits samples', x.shape)
    db = 'digits'
    n_clusters = 10
    # define and train SAE model
    sae = SAE(dims=[x.shape[-1], 64,32])
    sae.fit(x=x, epochs=400)
    sae.autoencoders.save_weights('weights_%s.pt' % db)

    # extract features
    print ('Finished training, extracting features using the trained SAE model')
    features = sae.extract_feature(x)
    print ('performing k-means clustering on the extracted features')
    km = KMeans(n_clusters, n_init=20)
    y_pred = km.fit_predict(features)
    print ('K-means clustering result on extracted features: NMI =', nmi(y, y_pred))
