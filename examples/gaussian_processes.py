import numpy as np
from tinygp import kernels

import genjax


console = genjax.go_pretty()

# Simulate a made up dataset, as an example
random = np.random.default_rng(1)
X = np.sort(random.uniform(0, 10, 10))
y = np.sin(X) + 1e-4 * random.normal(size=X.shape)

# Compute the log probability
kernel = 0.5 * kernels.ExpSquared(scale=1.0)
gp = genjax.GaussianProcess(kernel)
print(gp.logpdf(y, X, diag=1e-4))
