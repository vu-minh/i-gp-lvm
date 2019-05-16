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
import umap

from bokeh.palettes import Spectral10 as spectral_color  # Category10  # Accent
from bokeh.plotting import figure, output_file, show
from bokeh.models import BoxSelectTool

from bokeh.layouts import gridplot
from bokeh.palettes import Viridis3

import math


assert pyro.__version__.startswith("0.3.0")

pyro.enable_validation(True)
pyro.set_rng_seed(1)


def run_gplvm(y, informative_prior=True):
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
    if informative_prior:
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
    plt.savefig("./plots/gplvm_losses.png")

    # now the mean and std of X (in q(X) ~ p(X|y)) will be store in X_loc and X_scale
    # important: to get sample from q(X), set `mode` of `gplvm` to `guide`
    gplvm.mode = "guide"  # default: "model"
    X = gplvm.X_loc.detach().numpy()

    return X
    # viz(X, name="gplvm")
    # viz_bokeh(X, name=("gplvm_with_prior" if informative_prior
    #                    else "gplvm_non_informative_prior"))


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


def viz_bokeh_group(y):
    w = 500
    h = 300
    tooltips_bigger_font = """
        <div>
            <span style="font-size: 18px; font-weight: bold;">stage: $name</span>
        </div>
    """

    pca = PCA(n_components=2)
    Z1 = pca.fit_transform(y)
    p1 = figure(plot_width=w, plot_height=h, title="PCA",
                tooltips=tooltips_bigger_font)
    p1 = viz_bokeh(Z1, name="PCA", show_label=False, show_legend=True, p=p1)

    tsne = TSNE(perplexity=20)
    Z2 = tsne.fit_transform(y)
    p2 = figure(plot_width=w, plot_height=h, title="t-SNE_perp20",
                tooltips=tooltips_bigger_font)
    p2 = viz_bokeh(Z2, name="", show_label=False,
                   show_legend=False, p=p2)

    Z3 = umap.UMAP(n_neighbors=20).fit_transform(y)
    p3 = figure(plot_width=w, plot_height=h, title="UMAP_neighbors20",
                tooltips=tooltips_bigger_font)
    p3 = viz_bokeh(Z3, name="", show_label=False,
                   show_legend=False, p=p3)

    Z4 = run_gplvm(y.t(), informative_prior=False)
    p4 = figure(plot_width=w, plot_height=h,
                title="GP-LVM Non-informative prior")
    p4 = viz_bokeh(Z4, name="", show_label=False,
                   show_legend=False, p=p4)

    grid = gridplot([[p1, p2], [p3, p4]])
    output_file(f"./plots/qPCR_bokeh_group1.html")
    show(grid)


def viz_bokeh(X, name="", show_label=True, show_legend=True, p=None):

    labels = df.index.unique()
    tooltips_default = [
        ("stage", "$name"),
    ]
    tooltips_bigger_font = """
        <div>
            <span style="font-size: 18px; font-weight: bold;">stage: $name</span>
        </div>
    """

    if p is None:
        p = figure(
            plot_width=800,
            plot_height=600,
            tooltips=tooltips_bigger_font,
            title=f"{name} qPCR dataset, 48 genes",
        )
    p.add_tools(BoxSelectTool())

    for i, (label, color) in enumerate(zip(labels, spectral_color)):
        X_i = X[df.index == label]

        p.circle(X_i[:, 0], X_i[:, 1],
                 # radius=0.1,
                 size=10,
                 color=color,
                 alpha=0.6,
                 muted_color=color,
                 muted_alpha=0.2,
                 selection_fill_color=color,
                 selection_fill_alpha=1.0,
                 nonselection_fill_alpha=0.2,
                 legend=(label if show_legend else None), name=label)
    if show_legend:
        # p.legend.location = "top_left"
        p.legend.click_policy = "mute"  # "hide"

    if show_label:
        p.xaxis.axis_label = "pseudo-time"
        p.yaxis.axis_label = "branching"

    if p is None:
        output_file(f"./plots/qPCR_{name}_bokeh.html")
        show(p)
    else:
        return p


def run_tsne(y, perp=20):
    tsne = TSNE(perplexity=perp)
    Z = tsne.fit_transform(y)
    viz(Z, f"tsne_p{perp}", show_label=False)
    viz_bokeh(Z, f"tsne_p{perp}", show_label=False)


def run_pca(y):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(y)
    viz(Z, "pca", show_label=False)
    viz_bokeh(Z, "pca", show_label=False)


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
    # run_gplvm(y)
    # run_tsne(data, perp=20)
    # run_tsne(data, perp=10)
    # run_tsne(data, perp=35)
    # run_pca(data)

    viz_bokeh_group(data)
