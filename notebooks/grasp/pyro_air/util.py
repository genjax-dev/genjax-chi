import torch

from .air import latents_to_tensor


def count_accuracy(X, true_counts, air, batch_size):
    assert X.size(0) == true_counts.size(0), "Size mismatch."
    assert X.size(0) % batch_size == 0, "Input size must be multiple of batch_size."
    counts = torch.LongTensor(3, 4).zero_()
    error_latents = []
    error_indicators = []

    def count_vec_to_mat(vec, max_index):
        out = torch.LongTensor(vec.size(0), max_index + 1).zero_()
        out.scatter_(1, vec.type(torch.LongTensor).view(vec.size(0), 1), 1)
        return out

    for i in range(X.size(0) // batch_size):
        X_batch = X[i * batch_size : (i + 1) * batch_size]
        true_counts_batch = true_counts[i * batch_size : (i + 1) * batch_size]
        z_where, z_pres = air.guide(X_batch, batch_size)
        inferred_counts = sum(z.cpu() for z in z_pres).squeeze().data
        true_counts_m = count_vec_to_mat(true_counts_batch, 2)
        inferred_counts_m = count_vec_to_mat(inferred_counts, 3)
        counts += torch.mm(true_counts_m.t(), inferred_counts_m)
        error_ind = 1 - (true_counts_batch == inferred_counts).long()
        error_ix = error_ind.nonzero(as_tuple=False).squeeze()
        error_latents.append(
            latents_to_tensor((z_where, z_pres)).index_select(0, error_ix)
        )
        error_indicators.append(error_ind)

    acc = counts.diag().sum().float() / X.size(0)
    error_indices = torch.cat(error_indicators).nonzero(as_tuple=False).squeeze()
    if X.is_cuda:
        error_indices = error_indices.cuda()
    return acc, counts, torch.cat(error_latents), error_indices


# Defines something like a truncated geometric. Like the geometric,
# this has the property that there's a constant difference in log prob
# between p(steps=n) and p(steps=n+1).
def make_prior(k):
    assert 0 < k <= 1
    u = 1 / (1 + k + k**2 + k**3)
    p0 = 1 - u
    p1 = 1 - (k * u) / p0
    p2 = 1 - (k**2 * u) / (p0 * p1)
    trial_probs = [p0, p1, p2]
    # dist = [1 - p0, p0 * (1 - p1), p0 * p1 * (1 - p2), p0 * p1 * p2]
    # print(dist)
    return lambda t: trial_probs[t]


def is_baseline_param(param_name: str):
    return "bl_" in param_name


def get_per_param_lr(learning_rate: float, baseline_lr: float):
    def per_param_optim_args(param_name):
        lr = baseline_lr if is_baseline_param(param_name) else learning_rate
        return {"lr": lr}

    return per_param_optim_args
