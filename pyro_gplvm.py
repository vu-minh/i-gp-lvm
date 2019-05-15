"""Try to make GP-LVM model in pyro work.
    13/05/2019
    Ref: http://pyro.ai/examples/gplvm.html
"""

import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torch.nn import Parameter

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


assert pyro.__version__.startswith("0.3.0")

pyro.enable_validation(True)
pyro.set_rng_seed(1)


def run_gplvm(y):
    # the latent variables are X (in the tut, X is called Latent Space)
    # dim(X) = 2 to describe 2 aspects:
    #   + capture-time (1,2,4,8,32,64) (6 stages)
    #   + cell-branching types (TE, ICM, PE, EPI)

    # Stick the capture-time feature to x-axis
    # note that, we are using the supervised information
    capture_time = y.new_tensor(
        [int(cell_name.split(" ")[0]) for cell_name in df.index.values]
    )  # in [1, 2, 4, 8, 32, 64]

    capture_time_normalized = capture_time.log2() / 6  # in range [0, 1]

    # try to corrupt this supervised info, e.g., let keep 10% of them
    print(capture_time_normalized.shape)
    mask = torch.randint(
        low=0,
        high=capture_time_normalized.size(0),
        size=(int(0.9 * capture_time_normalized.size(0)),),
    )
    capture_time_normalized[mask] = -0.1

    # setup the mean of the prior over X
    X_prior_mean = torch.zeros(y.size(1), 2)  # n_observations x x_sim
    X_prior_mean[:, 0] = capture_time_normalized
    # note that X has 2 features
    # the first feature we set the prior to capture_time_normalized (this is just the prior)
    # this will be changed in the posterior
    # and the second features has zero mean, it will be inferred "from scratch"

    # construction of a Sparse Gaussian Process

    # RBF kernel
    kernel = gp.kernels.RBF(input_dim=2, lengthscale=torch.ones(2))

    # define X as Parameter so its "param" can be learned / we can set a prior and guide
    X = Parameter(X_prior_mean.clone())

    # build a SparesGP with num_inducing=32
    Xu = stats.resample(X_prior_mean.clone(), 32)
    gplvm = gp.models.SparseGPRegression(
        X, y, kernel, Xu=Xu, noise=torch.tensor(0.01), jitter=1e-5
    )

    # set prior and guide for the GP-LVM
    gplvm.set_prior("X", dist.Normal(X_prior_mean, 0.1).to_event())
    gplvm.autoguide("X", dist.Normal)

    # Inference: train GP by gp.util.train <- which use VI with Adam lr=0.01
    t = time.time()
    print("Start training")
    losses = gp.util.train(gplvm, num_steps=4000)
    print(f"Training GP-LVM in {time.time() - t} seconds")

    plt.plot(losses)
    plt.savefig("gplvm_losses.png")

    # now the mean and std of X (in q(X) ~ p(X|y)) will be store in X_loc and X_scale
    # important: to get sample from q(X), set `mode` of `gplvm` to `guide`
    gplvm.mode = "guide"  # default: "model"
    X = gplvm.X_loc.detach().numpy()

    viz(X, name="gplvm")


# viz X in 2D
def viz(X, name, show_label=True):
    plt.figure(figsize=(8, 6))
    colors = plt.get_cmap("tab10").colors[::-1]
    labels = df.index.unique()

    for i, label in enumerate(labels):
        # get each group
        X_i = X[df.index == label]
        plt.scatter(X_i[:, 0], X_i[:, 1], c=np.array([colors[i]]), label=label)

    plt.legend()
    if show_label:
        plt.xlabel("pseudo-time", fontsize=14)
        plt.ylabel("branching", fontsize=14)
    plt.title(name, fontsize=16)
    plt.savefig(f"./plots/qPCR_{name}.png")


def run_tsne(y, perp=20):
    tsne = TSNE(perplexity=perp)
    Z = tsne.fit_transform(y)
    viz(Z, f"tsne_p{perp}", show_label=False)


def run_pca(y):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(y)
    viz(Z, "pca", show_label=False)


if __name__ == "__main__":

    in_name = "./qPCR/guo_qpcr.csv"
    df = pd.read_csv(in_name, index_col=0)
    print(df.shape)
    print(df.index.unique().tolist())
    print(df.head())

    # note that the (measured) data is standardized for each feature (each gene)
    print(df.describe())

    # output tensor y
    # the number of Gaussian processes needed == number of genes (features)
    data = torch.tensor(df.values, dtype=torch.get_default_dtype())
    y = data.t()

    # each row of y is one gene
    print(y.shape)

    # test different DR methods
    run_gplvm(y)
    # run_tsne(data, perp=20)
    # run_tsne(data, perp=10)
    # run_tsne(data, perp=35)
    # run_pca(data)
