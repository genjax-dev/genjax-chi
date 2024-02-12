# Variational inference


<figure markdown="span">
  ![GenJAX VI architecture](../../assets/img/genjax-vi.png){ width = "300" }
  <figcaption>How variational inference works in GenJAX.</figcaption>
</figure>

**Variational inference** is an approximate inference technique where the problem of computing the intractable posterior distribution $P'$ is transformed into an optimization problem. The idea is to find a distribution $Q$ that is close to the true posterior $P'$ by minimizing the Kullback-Leibler (KL) divergence between the two distributions. 

::: genjax._src.inference.vi
    options:
      members:
        - ADEVDistribution
        - ExpectedValueLoss
        - ELBO
        - IWELBO
        - QWake
      show_root_heading: true
