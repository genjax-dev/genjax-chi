var __index = {"config":{"lang":["en"],"separator":"[\\s\\-]+","pipeline":["stopWordFilter"],"fields":{"title":{"boost":1000.0},"text":{"boost":1.0},"tags":{"boost":1000000.0}}},"docs":[{"location":"index.html","title":"Overview","text":"<p>GenJAX: a probabilistic programming library designed from the ground up to scale Bayesian modeling and inference into high performance settings. (1)</p> <ol> <li> <p>Here, high performance means massively parallel, either cores or devices.</p> <p>For those whom this overview page may be irrelevant: the value proposition is about putting expressive models and customizable Bayesian inference on GPUs, TPUs, etc - without sacrificing abstraction or modularity.</p> </li> </ol> <p>Gen is a multi-paradigm (generative, differentiable, incremental) system for probabilistic programming. GenJAX is an implementation of Gen on top of JAX (2) - exposing the ability to programmatically construct and manipulate generative functions (1) (computational objects which represent probability measures over structured sample spaces), with compilation to native devices, accelerators, and other parallel fabrics. </p> <ol> <li> <p>By design, generative functions expose a concise interface for expressing approximate and differentiable inference algorithms. </p> <p>The set of generative functions is extensible! You can implement your own - allowing advanced users to performance optimize their critical modeling/inference code paths.</p> <p>You can (and we, at the MIT Probabilistic Computing Project, do!) use these objects for machine learning - including robotics, natural language processing, reasoning about agents, and modelling / creating systems which exhibit human-like reasoning.</p> <p>A precise mathematical formulation of generative functions is given in Marco Cusumano-Towner's PhD thesis.</p> </li> <li> <p>If the usage of JAX is not a dead giveaway, GenJAX is written in Python.</p> </li> </ol> Model codeInference code <p><p> Defining a beta-bernoulli process in GenJAX. </p></p> <pre><code>@genjax.gen\ndef model():\np = Beta(0, 1) @ \"p\"\nv = Bernoulli(p) @ \"v\"\nreturn v\n</code></pre> <p><p> This works for any generative function, not just the beta-bernoulli model. </p></p> <pre><code># Sampling importance resampling.\ndef sir(prng_key: PRNGKey, gen_fn: GenerativeFunction\nobs: ChoiceMap, args: Tuple, n_samples: Int):\npass\n</code></pre>"},{"location":"index.html#what-sort-of-things-do-you-use-genjax-for","title":"What sort of things do you use GenJAX for?","text":"Real time object tracking <p>Real time tracking of objects in 3D using probabilistic rendering. (Left) Ground truth, (center) depth mask, (right) inference overlaid on ground truth.</p> <p><p> </p></p>"},{"location":"index.html#why-gen","title":"Why Gen?","text":"<p>GenJAX is a Gen implementation. If you're considering using GenJAX - it's worth starting by understanding why Gen exists, and what problems it purports to solve.</p>"},{"location":"index.html#the-evolution-of-probabilistic-programming-languages","title":"The evolution of probabilistic programming languages","text":"<p>Probabilistic modeling and inference is hard: understanding a domain well enough to construct a probabilistic model in the Bayesian paradigm is challenging, and that's half the battle - the other half is designing effective inference algorithms to probe the implications of the model (1).</p> <ol> <li> <p>Some probabilistic programming languages restrict the set of allowable models, providing (in return) efficient (often, exact) inference. </p> <p>Gen considers a wide class of models - include Bayesian nonparametrics, open-universe models, and models over rich structures (like programs!) - which don't natively support efficient exact inference.</p> </li> </ol> <p>In the past, probabilistic modellers typically considered the following design loop.</p> <pre><code>graph LR\n  A[Design model.] --&gt; B[Implement inference by hand.];\n  B --&gt; C[Model + inference okay?];\n  C --&gt; D[Happy.];\n  C --&gt; A;</code></pre> <p>The first generation (1) of probabilistic programming systems introduced inference engines which could operate abstractly over many different models, without requiring the programmer to return and tweak their inference code. The utopia envisioned by these systems is shown below.</p> <ol> <li> <p>Here, the definition of \"first generation\" includes systems like JAGS, BUGS, BLOG, IBAL, Church, Infer.NET, Figaro, Stan, amongst others.</p> <p>But more precisely, many systems preceded the DARPA PPAML project - which gave rise to several novel systems, including the predecessors of Gen.</p> </li> </ol> <pre><code>graph LR\n  A[Design model.] --&gt; D[Model + inference okay?];\n  B[Inference engine.] ---&gt; D;\n  D --&gt; E[Happy.];\n  D ---&gt; A;</code></pre> <p>The problem with this utopia is that we often do need to program our inference algorithms (1) to achieve maximum performance, with respect to accuracy as well as runtime. First generation systems were not designed with this in mind.</p> <ol> <li>Here, programmable inference denotes using a custom proposal distribution in importance sampling, or a custom variational family for variational inference, or even a custom kernel in Markov chain Monte Carlo.</li> </ol>"},{"location":"index.html#programmable-inference","title":"Programmable inference","text":"<p>The goal then is to allow users to customize when required, while retaining the rapid model/inference iteration properties explored by first generation systems.</p> <p>Gen addresses this goal by introducing a separation between modeling and inference code: the generative function interface.</p> <p> </p> <p>The interface provides an abstraction layer that inference algorithms can call to compute the right math (think: importance weights, accept reject ratios, gradient estimators). Advanced developers can create new model languages by implementing the interface - and immediately gain access to advanced inference procedures.</p>"},{"location":"index.html#whose-using-gen","title":"Whose using Gen?","text":"<p>Gen supports a growing list of users, with collaboration across academic research labs and industry affiliates.</p> <p> </p> <p>We're looking to expand our user base! If you're interested, please contact us to get involved.</p>"},{"location":"genjax/notebooks.html","title":"Modeling &amp; inference notebooks","text":"<p>Link to the notebook repository</p> <p>This section contains a link to a (statically hosted) series of tutorial notebooks designed to guide usage of GenJAX. These notebooks are executed and rendered with quarto, and are kept up to date with the repository along with the documentation.</p> <p>The notebook repository can be found here.</p>"},{"location":"genjax/concepts/diff_jl.html","title":"Diffing against Gen.jl","text":"<p><code>GenJAX</code> is inherits concepts from Gen and algorithm reference implementations from <code>Gen.jl</code> - there are a few necessary design deviations between <code>GenJAX</code> and <code>Gen.jl</code> that stem from JAX's underlying array programming model. In this section, we describe several of these differences and try to highlight workarounds or discuss the reason for the discrepancy.</p>"},{"location":"genjax/concepts/diff_jl.html#turing-universality","title":"Turing universality","text":"<p><code>Gen.jl</code> is Turing universal - it can encode any computable distribution, including those expressed by forms of unbounded recursion.</p> <p>It is a bit ambiguous whether or not <code>GenJAX</code> falls in this category: JAX does not feature mechanisms for dynamic shape allocations, but it does feature mechanisms for unbounded recursion.</p> <p>The former provides a technical barrier to implementing Gen's trace machinery. While JAX allows for unbounded recursion, to support Gen's interfaces we also need the ability to dynamically allocate choice data. This requirement is currently at tension with XLA's requirements of knowing the static shape of everything.</p> <p>However, <code>GenJAX</code> supports generative function combinators with bounded recursion / unfold chain length. Ahead of time, these combinators can be directed to pre-allocate arrays with enough size to handle recursion/looping within the bounds that the programmer sets. If these bounds are exceeded, a Python runtime error will be thrown (both on and off JAX device).</p> <p>In practice, this means that some performance engineering (space vs. expressivity) is required of the programmer. It's certainly feasible to express bounded recursive computations which terminate with probability 1 - but you'll need to ahead of time allocate space for it.</p>"},{"location":"genjax/concepts/diff_jl.html#mutation","title":"Mutation","text":"<p>Just like JAX, GenJAX disallows mutation - expressing a mutation to an array must be done through special interfaces, and those interfaces return full copies. There are special circumstances where these interfaces will be performed in place.</p>"},{"location":"genjax/concepts/diff_jl.html#to-jit-or-not-to-jit","title":"To JIT or not to JIT","text":"<p><code>Gen.jl</code> is written in Julia, which automatically JITs everything. <code>GenJAX</code>, by virtue of being constructed on top of JAX, allows us to JIT JAX compatible code - but the JIT process is user directed. Thus, the idioms that are used to express and optimize inference code are necessarily different compared to <code>Gen.jl</code>. In the inference standard library, you'll typically find algorithms implemented as dataclasses which inherit (and implement) the <code>jax.Pytree</code> interfaces. Implementing these interfaces allow usage of inference dataclasses and methods in jittable code - and, as a bonus, allow us to be specific about trace vs. runtime known values.</p> <p>In general, it's productive to enclose as much of a computation as possible in a <code>jax.jit</code> block. This can sometimes lead to long trace times. If trace times are ballooning, a common source is explicit for-loops (with known bounds, else JAX will complain). In these cases, you might look at Advice on speeding up compilation time. We've taken care to optimize (by e.g. using XLA primitives) the code which we expose from GenJAX - but if you find something out of the ordinary, file an issue!</p>"},{"location":"genjax/concepts/generative_functions.html","title":"Generative functions","text":"<p>Gen is all about generative functions: computational objects which support an interface that helps automate the tricky math involved in programming Bayesian inference algorithms. In this section, we'll unpack the generative function interface and explain the mathematics behind generative functions (1).</p> <ol> <li>For a deeper dive, enjoy Marco Cusumano-Towner's PhD thesis.</li> </ol>"},{"location":"genjax/library/core.html","title":"Core","text":"<p>This module provides the core functionality and JAX compatibility layer which the <code>GenJAX</code> generative function modeling and inference modules are built on top of. It contains (truncated, and in no particular order):</p> <ul> <li> <p>Core Gen associated data types for generative functions.</p> </li> <li> <p>Utility functionality for automatically registering class definitions as valid <code>Pytree</code> method implementors (guaranteeing <code>flatten</code>/<code>unflatten</code> compatibility across JAX transform boundaries). For more information, see Pytrees.</p> </li> <li> <p>Staging functionality that allows linear lifting of pure, numerical Python programs to <code>ClosedJaxpr</code> instances.</p> </li> <li> <p>Transformation interpreters: interpreter-based transformations on which operate on <code>ClosedJaxpr</code> instances. Interpreters are all written in initial style - they operate on <code>ClosedJaxpr</code> instances, and don't implement their own custom <code>jax.Tracer</code> types - but they are JAX compatible, implying that they can be staged out for zero runtime cost.</p> </li> <li> <p>Masking functionality which allows active/inactive flagging of data - useful when branching patterns of computation require uncertainty in whether or not data is active with respect to a generative computation.</p> </li> </ul>"},{"location":"genjax/library/core.html#genjax.core.GenerativeFunction","title":"<code>genjax.core.GenerativeFunction</code>  <code>dataclass</code>","text":"<p>         Bases: <code>Pytree</code></p> <p>Abstract class which provides an inheritance base for user-defined implementations of the generative function interface methods e.g. the <code>BuiltinGenerativeFunction</code> and <code>Distribution</code> languages both implement a class inheritor of <code>GenerativeFunction</code>.</p> <p>Any implementation will interact with the JAX tracing machinery, however, so there are specific API requirements above the requirements enforced in other languages (unlike Gen in Julia, for example).</p> <p>The user must match the interface signatures of the native JAX implementation. This is not statically checked - but failure to do so will lead to unintended behavior or errors.</p> <p>To support argument and choice gradients via JAX, the user must provide a differentiable <code>importance</code> implementation.</p> Source code in <code>genjax/_src/core/datatypes/generative.py</code> <pre><code>@dataclasses.dataclass\nclass GenerativeFunction(Pytree):\n\"\"\"Abstract class which provides an inheritance base for user-defined\n    implementations of the generative function interface methods e.g. the\n    `BuiltinGenerativeFunction` and `Distribution` languages both implement a\n    class inheritor of `GenerativeFunction`.\n    Any implementation will interact with the JAX tracing machinery,\n    however, so there are specific API requirements above the requirements\n    enforced in other languages (unlike Gen in Julia, for example).\n    The user *must* match the interface signatures of the native JAX\n    implementation. This is not statically checked - but failure to do so\n    will lead to unintended behavior or errors.\n    To support argument and choice gradients via JAX, the user must\n    provide a differentiable `importance` implementation.\n    \"\"\"\n# This is used to support tracing -- the user is not required to provide\n# a PRNGKey, because the value of the key is not important, only\n# the fact that the value has type PRNGKey.\ndef __abstract_call__(self, *args) -&gt; Tuple[PRNGKey, Any]:\nkey = jax.random.PRNGKey(0)\n_, tr = self.simulate(key, args)\nretval = tr.get_retval()\nreturn retval\ndef get_trace_type(self, *args, **kwargs) -&gt; TraceType:\nshape = kwargs.get(\"shape\", ())\nreturn Bottom(shape)\n@abc.abstractmethod\ndef simulate(\nself,\nkey: PRNGKey,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Trace]:\npass\n@abc.abstractmethod\ndef importance(\nself,\nkey: PRNGKey,\nchm: ChoiceMap,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[FloatArray, Trace]]:\npass\n@abc.abstractmethod\ndef update(\nself,\nkey: PRNGKey,\noriginal: Trace,\nnew: ChoiceMap,\ndiffs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[Any, FloatArray, Trace, ChoiceMap]]:\npass\n@abc.abstractmethod\ndef assess(\nself,\nkey: PRNGKey,\nevaluation_point: ChoiceMap,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[Any, FloatArray]]:\npass\ndef unzip(\nself,\nkey: PRNGKey,\nfixed: ChoiceMap,\n) -&gt; Tuple[\nPRNGKey,\nCallable[[ChoiceMap, Tuple], FloatArray],\nCallable[[ChoiceMap, Tuple], Any],\n]:\nkey, sub_key = jax.random.split(key)\ndef score(differentiable: Tuple, nondifferentiable: Tuple) -&gt; FloatArray:\nprovided, args = tree_zipper(differentiable, nondifferentiable)\nmerged = fixed.merge(provided)\n_, (_, score) = self.assess(sub_key, merged, args)\nreturn score\ndef retval(differentiable: Tuple, nondifferentiable: Tuple) -&gt; Any:\nprovided, args = tree_zipper(differentiable, nondifferentiable)\nmerged = fixed.merge(provided)\n_, (retval, _) = self.assess(sub_key, merged, args)\nreturn retval\nreturn key, score, retval\n# A higher-level gradient API - it relies upon `unzip`,\n# but provides convenient access to first-order gradients.\ndef choice_grad(self, key, trace, selection):\nfixed = selection.complement().filter(trace.strip())\nevaluation_point = selection.filter(trace.strip())\nkey, scorer, _ = self.unzip(key, fixed)\ngrad, nograd = tree_grad_split(\n(evaluation_point, trace.get_args()),\n)\nchoice_gradient_tree, _ = jax.grad(scorer)(grad, nograd)\nreturn key, choice_gradient_tree\n</code></pre>"},{"location":"genjax/library/core.html#genjax._src.core.datatypes.generative.GenerativeFunction.simulate","title":"<code>simulate(key, args)</code>  <code>abstractmethod</code>","text":"Source code in <code>genjax/_src/core/datatypes/generative.py</code> <pre><code>@abc.abstractmethod\ndef simulate(\nself,\nkey: PRNGKey,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Trace]:\npass\n</code></pre>"},{"location":"genjax/library/core.html#genjax._src.core.datatypes.generative.GenerativeFunction.importance","title":"<code>importance(key, chm, args)</code>  <code>abstractmethod</code>","text":"Source code in <code>genjax/_src/core/datatypes/generative.py</code> <pre><code>@abc.abstractmethod\ndef importance(\nself,\nkey: PRNGKey,\nchm: ChoiceMap,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[FloatArray, Trace]]:\npass\n</code></pre>"},{"location":"genjax/library/core.html#genjax._src.core.datatypes.generative.GenerativeFunction.update","title":"<code>update(key, original, new, diffs)</code>  <code>abstractmethod</code>","text":"Source code in <code>genjax/_src/core/datatypes/generative.py</code> <pre><code>@abc.abstractmethod\ndef update(\nself,\nkey: PRNGKey,\noriginal: Trace,\nnew: ChoiceMap,\ndiffs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[Any, FloatArray, Trace, ChoiceMap]]:\npass\n</code></pre>"},{"location":"genjax/library/core.html#genjax._src.core.datatypes.generative.GenerativeFunction.assess","title":"<code>assess(key, evaluation_point, args)</code>  <code>abstractmethod</code>","text":"Source code in <code>genjax/_src/core/datatypes/generative.py</code> <pre><code>@abc.abstractmethod\ndef assess(\nself,\nkey: PRNGKey,\nevaluation_point: ChoiceMap,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[Any, FloatArray]]:\npass\n</code></pre>"},{"location":"genjax/library/generative_functions.html","title":"Generative functions","text":"<p>This module contains several standard generative function classes useful for structuring probabilistic programs.</p>"},{"location":"genjax/library/inference.html","title":"Inference","text":""}]}