"""
PyTorch implementation of "Deep learning enables accurate clustering and batch effect removal in single-cell RNA-seq analysis"
"""
import math
import os
import random
from time import time as get_time

import igraph
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch import nn

from .SAE import SAE
from ._nn import (
    SGD,
    Dense,
    Dropout,
    EarlyStopping,
    Sequential,
    as_float32_array,
    fit,
    get_loss,
    get_optimizer,
    init_kernel_,
    set_seed,
    train_on_batch,
)


class ClusteringLayer(nn.Module):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        clustering = ClusteringLayer(n_clusters=10, input_dim=32)
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
        input_dim: number of features. Can be omitted when `weights` is given.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, input_dim=None, name='clustering'):
        super().__init__()
        if input_dim is None:
            if weights is None:
                raise ValueError('ClusteringLayer needs either `input_dim` or initial `weights`')
            input_dim = np.asarray(weights[0]).shape[1]
        self.name = name
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_dim = input_dim
        self.clusters = nn.Parameter(torch.empty(n_clusters, input_dim))
        init_kernel_(self.clusters, 'glorot_uniform')
        if weights is not None:
            self.set_weights(weights)

    def forward(self, inputs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (torch.sum(torch.square(inputs.unsqueeze(1) - self.clusters), dim=2) / self.alpha))
        q = q ** ((self.alpha + 1.0) / 2.0)
        return q / torch.sum(q, dim=1, keepdim=True)

    def get_weights(self):
        return [self.clusters.detach().cpu().numpy().copy()]

    def set_weights(self, weights):
        with torch.no_grad():
            self.clusters.copy_(torch.as_tensor(np.asarray(weights[0])))

    def get_config(self):
        return {'n_clusters': self.n_clusters, 'alpha': self.alpha, 'input_dim': self.input_dim, 'name': self.name}

    def output_dim(self):
        return self.n_clusters

    def extra_repr(self):
        return 'n_clusters=%d, input_dim=%d, alpha=%g' % (self.n_clusters, self.input_dim, self.alpha)


class ClusteringLayerGaussian(ClusteringLayer):
    def forward(self, inputs):
        sigma=1.0
        q = torch.sum(torch.exp(-torch.square(inputs.unsqueeze(1) - self.clusters) / (2.0*sigma*sigma)), dim=2)
        return q / torch.sum(q, dim=1, keepdim=True)


_LAYER_CLASSES = {cls.__name__: cls for cls in (Dense, Dropout, ClusteringLayer, ClusteringLayerGaussian)}


def load_model(path, device=None):
    """Rebuild a model saved by `save()`, e.g. `autoencoder_model.pt` or `encoder_model.pt` in `save_dir`."""
    saved = torch.load(path, map_location=device or 'cpu', weights_only=True)
    model = Sequential([_LAYER_CLASSES[cls](**config) for cls, config in saved['layers']], name=saved['name'])
    model.load_state_dict(saved['state_dict'])
    return model.to(device or 'cpu')


def louvain(adjacency, resolution=1.0, random_state=0):
    """
    Louvain community detection on a weighted (kNN) graph, e.g. `adata.obsp['connectivities']`.

    DESC 2.x used `scanpy.tl.louvain`, backed by the `louvain` package, which cannot be installed
    on current Python versions. igraph's multilevel algorithm optimizes the same resolution-weighted
    modularity. Clusters are labelled by decreasing size, as in the `louvain` package.
    """
    adjacency = sp.csr_matrix(adjacency)
    edges = sp.triu(adjacency.maximum(adjacency.T), k=1).tocoo()
    graph = igraph.Graph(n=adjacency.shape[0], edges=np.column_stack((edges.row, edges.col)).tolist())
    # igraph draws random numbers from Python's `random` module unless told otherwise
    igraph.set_random_number_generator(random.Random(random_state))
    try:
        membership = np.asarray(graph.community_multilevel(weights=edges.data.astype(float).tolist(), resolution=resolution).membership)
    finally:
        igraph.set_random_number_generator(random)
    relabel = np.empty(membership.max() + 1, dtype=int)
    relabel[np.argsort(-np.bincount(membership), kind='stable')] = np.arange(relabel.shape[0])
    return relabel[membership]


class DescModel(object):
    def __init__(self,
                 dims,
                 x, # input matrix, row sample, col predictors
                 alpha=1.0,
                 tol=0.005,
                 init='glorot_uniform', #initialization method
                 louvain_resolution=1.0, # resolution for louvain
                 n_neighbors=10,    # the
                 pretrain_epochs=300, # epoch for autoencoder
                 epochs_fit=4, #epochs for each update,int or float
                 batch_size=256, #batch_size for autoencoder
                 random_seed=201809,
                 activation='relu',
                 actincenter="tanh",# activation for the last layer in encoder, and first layer in the decoder
                 drop_rate_SAE=0.2,
                 is_stacked=True,
                 use_earlyStop=True,
                 use_ae_weights=False,
                 save_encoder_weights=False,
                 save_encoder_step=5,
                 save_dir="result_tmp",
                 kernel_clustering="t",
                 # save result to save_dir, the default is "result_tmp". Missing parent directories are created.
                 device=None # torch device used for training, default CPU
                 ):

        if not os.path.exists(save_dir):
            print("Create the directory:"+str(save_dir)+" to save result")
            os.makedirs(save_dir)
        self.dims = dims
        self.x=as_float32_array(x) #feature n*p, n:number of cells, p: number of genes
        self.alpha = alpha
        self.tol=tol
        self.init=init
        self.input_dim = dims[0]  # for clustering layer
        self.n_stacks = len(self.dims) - 1
        self.is_stacked=is_stacked
        self.resolution=louvain_resolution
        self.n_neighbors=n_neighbors
        self.pretrain_epochs=pretrain_epochs
        self.epochs_fit=epochs_fit
        self.batch_size=batch_size
        self.random_seed=random_seed
        self.activation=activation
        self.actincenter=actincenter
        self.drop_rate_SAE=drop_rate_SAE
        self.use_earlyStop=use_earlyStop
        self.use_ae_weights=use_ae_weights
        self.save_encoder_weights=save_encoder_weights
        self.save_encoder_step=save_encoder_step
        self.save_dir=save_dir
        self.kernel_clustering=kernel_clustering
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.optimizer = None
        self.loss = None
        #set random seed
        set_seed(random_seed)
        #pretrain autoencoder
        self.pretrain()


    def pretrain(self):
        sae=SAE(dims=self.dims,
                act=self.activation,
                drop_rate=self.drop_rate_SAE,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                actincenter=self.actincenter,
                init=self.init,
                use_earlyStop=self.use_earlyStop,
                save_dir=self.save_dir,
                device=self.device
           )
        # begin pretraining
        t0 = get_time()
        ae_weights = os.path.join(self.save_dir, 'ae_weights.pt')
        keras_ae_weights = os.path.join(self.save_dir, 'ae_weights.h5')  # written by desc 2.x
        if self.use_ae_weights:
            print("Checking whether %s  exists in the directory"%str(ae_weights))
            if os.path.isfile(ae_weights):
                sae.autoencoders.load_weights(ae_weights)
            elif os.path.isfile(keras_ae_weights):
                print("Loading the Keras weights %s saved by desc 2.x" % keras_ae_weights)
                sae.autoencoders.load_weights(keras_ae_weights)
            else:
                print("The file ae_weights.pt is not exits")
                if self.is_stacked:
                    sae.fit(self.x,epochs=self.pretrain_epochs)
                else:
                    sae.fit2(self.x,epochs=self.pretrain_epochs)
        else:
            print("use_ae_weights=False, the program will rerun autoencoder")
            if self.is_stacked:
                sae.fit(self.x,epochs=self.pretrain_epochs)
            else:
                sae.fit2(self.x,epochs=self.pretrain_epochs)
        self.autoencoder=sae.autoencoders
        self.encoder=sae.encoder

        print('Pretraining time is', get_time() - t0)
        #save ae results into disk
        if not os.path.isfile(ae_weights):
            self.autoencoder.save_weights(ae_weights)
            self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights.pt'))
            print('Pretrained weights are saved to %s/ae_weights.pt' % self.save_dir)
        #save autoencoder model
        self.autoencoder.save(os.path.join(self.save_dir,"autoencoder_model.pt"))
        #initialize cluster centers using louvain
        features=self.extract_features(self.x)
        print("...number of clusters is unknown, Initialize cluster centroid using louvain method")
        #can be replaced by other clustering methods
        adata0=sc.AnnData(features)
        if adata0.shape[0]>200000:
            np.random.seed(adata0.shape[0])#set  seed
            adata0=adata0[np.random.choice(adata0.shape[0],200000,replace=False)].copy()
        num_threads=torch.get_num_threads()
        sc.pp.neighbors(adata0, n_neighbors=self.n_neighbors,use_rep="X")
        torch.set_num_threads(num_threads) # scanpy may reset the OpenMP thread count shared with torch
        self.init_pred=louvain(adata0.obsp['connectivities'],resolution=self.resolution)
        if np.unique(self.init_pred).shape[0]<=1:
            #avoid only a cluster
            raise ValueError("There is only a cluster detected. The resolution:"+str(self.resolution)+" is too small, please choose a larger resolution!!")
        features=pd.DataFrame(adata0.X,index=np.arange(0,adata0.shape[0]))
        Group=pd.Series(self.init_pred,index=np.arange(0,adata0.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean(),dtype=np.float32)
        self.n_clusters=cluster_centers.shape[0]
        self.init_centroid=[cluster_centers]
        #create desc clustering layer
        if self.kernel_clustering=="gaussian":
            clustering_layer = ClusteringLayerGaussian(self.n_clusters,weights=self.init_centroid,alpha=self.alpha,name='clustering')
        else:
            clustering_layer = ClusteringLayer(self.n_clusters,weights=self.init_centroid,alpha=self.alpha,name='clustering')
        self.model = Sequential(list(self.encoder.layers.values()) + [clustering_layer], name="desc").to(self.device)



    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        """
        optimizer: 'sgd', `desc.models.network.SGD(learning_rate, momentum)`, a `torch.optim.Optimizer`
            built on `self.model.parameters()`, or a callable taking the parameters and returning an optimizer.
        loss: 'kld', 'mse', or a callable loss(y_true, y_pred) on torch tensors.
        """
        self.optimizer = get_optimizer(optimizer, self.model.parameters())
        self.loss = get_loss(loss)

    def _check_compiled(self):
        if self.optimizer is None:
            raise RuntimeError('You must compile the model before training, e.g. desc.compile(optimizer=SGD(0.01, 0.9), loss="kld")')

    def fit_on_batch(self,maxiter=1e4,update_interval=200,save_encoder_step=4):
        self._check_compiled()
        #step1 initial weights by louvain,or Kmeans
        self.model.get_layer(name='clustering').set_weights(self.init_centroid)
        # Step 2: deep clustering
        y_pred_last = np.copy(self.init_pred)
        index_array = np.arange(self.x.shape[0])
        index=0
        for ite in range(int(maxiter)):
            if self.save_encoder_weights and ite%(save_encoder_step*update_interval)==0:
                self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights_resolution_'+str(self.resolution)+"_"+str(ite)+'.pt'))
                print('Fine tuning encoder weights are saved to %s/encoder_weights.pt' % self.save_dir)
            if ite % update_interval ==0:
                q=self.model.predict(self.x,verbose=0)
                p=torch.as_tensor(self.target_distribution(q),device=self.device)
                y_pred=q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                print("The value of delta_label of current",str(ite+1),"th iteration is",delta_label,">= tol",self.tol)
                if ite > 0 and delta_label < self.tol:
                    print('delta_label ', delta_label, '< tol ', self.tol)
                    print('Reached tolerance threshold. Stop training.')
                    break
            idx=index_array[index * self.batch_size: min((index+1) * self.batch_size, self.x.shape[0])]
            loss = train_on_batch(self.model, self.optimizer, self.loss, torch.as_tensor(self.x[idx],device=self.device), p[idx])
            index = index + 1 if (index + 1) * self.batch_size <= self.x.shape[0] else 0
        #save encoder model
        self.encoder.save(os.path.join(self.save_dir,"encoder_model.pt"))
        #load model
        #encoder=desc.models.network.load_model("encoder_model.pt")
        #
        y0=pd.Series(y_pred,dtype='category')
        y0=y0.cat.rename_categories(range(0,len(y0.cat.categories)))
        print("The final prediction cluster is:")
        x=y0.value_counts()
        print(x.sort_index(ascending=True))
        Embedded_z=self.extract_features(self.x)
        q=self.model.predict(self.x,verbose=0)
        return Embedded_z,q



    def fit_on_all(self, maxiter=1e3, epochs_fit=5,save_encoder_step=5): # unsupervised
        self._check_compiled()
        #step1 initial weights by louvain,or Kmeans
        self.model.get_layer(name='clustering').set_weights(self.init_centroid)
        # Step 2: deep clustering
        y_pred_last = np.copy(self.init_pred)
        for ite in range(int(maxiter)):
            if self.save_encoder_weights and ite%save_encoder_step==0: #save ae_weights for every 5 iterations
                self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights_resolution_'+str(self.resolution)+"_"+str(ite)+'.pt'))
                print('Fine tuning encoder weights are saved to %s/encoder_weights.pt' % self.save_dir)
            q = self.model.predict(self.x, verbose=0)
            p = self.target_distribution(q)  # update the auxiliary target distribution p
            # evaluate the clustering performance
            y_pred = q.argmax(1)
             # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < self.tol:
                print('delta_label ', delta_label, '< tol ', self.tol)
                print('Reached tolerance threshold. Stop training.')
                break
            print("The value of delta_label of current",str(ite+1),"th iteration is",delta_label,">= tol",self.tol)
            #train on whole dataset on prespecified batch_size
            early_stopping = EarlyStopping(min_delta=1e-4, patience=5) if self.use_earlyStop else None
            fit(self.model, self.x, p, self.optimizer, self.loss, batch_size=self.batch_size, epochs=epochs_fit, shuffle=True, early_stopping=early_stopping)
        #save encoder model
        self.encoder.save(os.path.join(self.save_dir,"encoder_model.pt"))
        #load model
        #encoder=desc.models.network.load_model("encoder_model.pt")
        #

        y0=pd.Series(y_pred,dtype='category')
        y0=y0.cat.rename_categories(range(0,len(y0.cat.categories)))
        print("The final prediction cluster is:")
        x=y0.value_counts()
        print(x.sort_index(ascending=True))
        Embedded_z=self.extract_features(self.x)
        return Embedded_z,q

    def fit(self,maxiter=1e4):
        if isinstance(self.epochs_fit,int):
            embedded_z,q=self.fit_on_all(maxiter=maxiter,epochs_fit=self.epochs_fit,save_encoder_step=self.save_encoder_step)
        else:
            update_interval=math.ceil(self.epochs_fit*self.x.shape[0]/self.batch_size)
            embedded_z,q=self.fit_on_batch(maxiter=maxiter,save_encoder_step=self.save_encoder_step,update_interval=update_interval)
        return embedded_z,q


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='DescModel class test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--tol', default=0.005, type=float)
    parser.add_argument('--save_dir', default='result_tmp')
    args = parser.parse_args()
    print(args)
    os.makedirs(args.save_dir, exist_ok=True)

    # use the scikit-learn digits data to test
    from sklearn.datasets import load_digits
    x, y = load_digits(return_X_y=True)
    x = x / 16.0
    print ('digits samples', x.shape)
    dims=[x.shape[-1],64,32]
    # prepare sample data to  the DESC model
    desc = DescModel(dims=dims,x=x,louvain_resolution=0.8,use_ae_weights=True,epochs_fit=0.4,pretrain_epochs=args.pretrain_epochs,batch_size=args.batch_size,tol=args.tol,save_dir=args.save_dir)
    desc.model.summary()
    t0 = get_time()
    desc.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    Embedded_z,q_pred= desc.fit(maxiter=30)
    y_pred=q_pred.argmax(axis=1)
    obs_info=pd.DataFrame()
    obs_info["y_true"]=pd.Series(y.astype("U"),dtype="category")
    obs_info["y_pred"]=pd.Series(y_pred.astype("U"),dtype="category")
    adata=sc.AnnData(x,obs=obs_info)
    adata.obsm["X_Embeded_z"]=Embedded_z
    print('clustering time: ', (get_time() - t0))
