from pathlib import Path
from typing import Tuple

import numpy as np
import pyro
import pyro.contrib.examples.multi_mnist as multi_mnist
import pytest
import torch
from pyro.infer import ELBO
from pyro.infer import SVI
from pyro.infer import RenyiELBO
from pyro.infer import TraceGraph_ELBO
from pyro.optim import Adam
from pyro_air import AIR
from pyro_air import get_per_param_lr
from pyro_air import make_prior


#####################
# Benchmark Configs
#####################
seed = 123456
use_cuda = torch.cuda.is_available()
batch_size = 64
num_epoches = 1

z_pres_prior = 0.01
learning_rate = 1e-4
baseline_lr = 1e-1
# explicitly list out all configurable options
air_model_args = dict(
    num_steps=3,
    x_size=50,
    window_size=28,
    z_what_size=50,
    rnn_hidden_size=256,
    encoder_net=[200],
    decoder_net=[200],
    predict_net=[200],
    embed_net=None,
    bl_predict_net=[200],
    non_linearity="ReLU",
    decoder_output_bias=-2,
    decoder_output_use_sigmoid=True,
    use_masking=True,
    use_baselines=True,
    baseline_scalar=None,
    scale_prior_mean=3.0,
    scale_prior_sd=0.2,
    pos_prior_mean=0.0,
    pos_prior_sd=1.0,
    likelihood_sd=0.3,
    use_cuda=use_cuda,
)

#####################
# Initial Setup
#####################
device = torch.device("cuda" if use_cuda else "cpu")
z_pres_prior_fn = make_prior(z_pres_prior)


# Load data once for all tests
@pytest.fixture(scope="session")
def multi_mnist_data():
    X, Y = multi_mnist.load(Path(__file__).parent / "data/air/.data")
    X = torch.from_numpy(X).float() / 255.0
    # Using float to allow comparison with values sampled from
    # Bernoulli.
    counts = torch.tensor([len(objs) for objs in Y], dtype=torch.float)
    return X.to(device), counts


@pytest.fixture(autouse=True)
def setup():
    if seed is not None:
        pyro.set_rng_seed(seed)
    pyro.distributions.enable_validation(False)
    pyro.clear_param_store()  # just in case


def train_air(svi: SVI, air: AIR, multi_mnist_data: Tuple[torch.Tensor, torch.Tensor]):
    data, true_counts = multi_mnist_data
    num_steps = int(np.ceil((data.size(0) / batch_size) * num_epoches))

    for _ in range(1, num_steps + 1):
        loss = svi.step(data, batch_size=batch_size, z_pres_prior_p=z_pres_prior_fn)
        # # evaluate count accuracy
        # accuracy, counts, error_z, error_ix = count_accuracy(data, true_counts, air, 1000)


@pytest.mark.parametrize("elbo", [TraceGraph_ELBO(), RenyiELBO(alpha=0)], ids=type)
def test_benchmark(
    benchmark, elbo: ELBO, multi_mnist_data: Tuple[torch.Tensor, torch.Tensor]
):
    air = AIR(**air_model_args)
    adam = Adam(get_per_param_lr(learning_rate, baseline_lr))
    svi = SVI(air.model, air.guide, adam, loss=elbo)

    benchmark(train_air, svi, air, multi_mnist_data)
