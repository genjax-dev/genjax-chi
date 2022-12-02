# Writing your first inference algorithm

Let's consider a generalized version of regression where we wish to model a data generating process using polynomials with outlier draws and noise. One such model is given below.

```{jupyter-execute}
import jax
import genjax
```
