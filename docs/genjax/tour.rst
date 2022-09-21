A tour of fundamentals
======================

It's useful to start with a dead simple modeling example (often the first example in many probabilistic programming frameworks) called *eight schools*. 

The *Eight schools* problem is covered extensively in Gelman et al., Bayesian Data Analysis: Sec. 5.5, 2003. 

.. admonition:: Eight schools

  A study was performed for the Educational Testing Service to analyze the effects of special coaching programs for SAT-V (Scholastic Aptitude Test-Verbal) in each of eight high schools. The outcome variable in each study was the score on a special administration of the SAT-V, a standardized multiple choice test administered by the Educational Testing Service and used to help colleges make admissions decisions; the scores can vary between 200 and 800, with mean about 500 and standard deviation about 100. The SAT examinations are designed to be resistant to short-term efforts directed specifically toward improving performance on the test; instead they are designed to reflect knowledge acquired and abilities developed over many years of education. Nevertheless, each of the eight schools in this study considered its short-term coaching program to be very successful at increasing SAT scores. Also, there was no prior reason to believe that any of the eight programs was more effective than any other or that some were more similar in effect to each other than to any other.

This example is also covered in `NumPyro`_ - so it is useful to compare the modeling and inference idioms in the context of something which is well-known.

.. jupyter-execute::
    
    # Eight schools in idiomatic GenJAX.

    import jax
    import jax.numpy as jnp
    import genjax


    # Create a MapCombinator generative function, mapping over
    # the key and sigma arguments.
    @genjax.gen(genjax.MapCombinator, in_axes=(0, None, None, 0))
    def plate(key, mu, tau, sigma):
        key, theta = genjax.trace("theta", genjax.Normal)(key, (mu, tau))
        key, obs = genjax.trace("obs", genjax.Normal)(key, (theta, sigma))
        return key, obs

    @genjax.gen
    def J_schools(key, J, sigma):
        key, mu = genjax.trace("mu", genjax.Normal)(key, (0.0, 5.0))
        key, tau = genjax.trace("tau", genjax.Cauchy)(key, ())
        key, *subkeys = jax.random.split(key, J + 1)
        subkeys = jnp.array(subkeys)
        _, obs = genjax.trace("plate", plate)(subkeys, (mu, tau, sigma))
        return key, obs


    # If one ever needs to specialize on arguments, you can just 
    # pass a lambda which closes over constants into 
    # a `BuiltinGenerativeFunction`.
    #
    # Here, we specialize on the number of schools.
    eight_schools = genjax.BuiltinGenerativeFunction(
        lambda key, sigma: J_schools(key, 8, sigma)
    )

.. _NumPyro: https://github.com/pyro-ppl/numpyro
