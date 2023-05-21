var __index = {"config":{"lang":["en"],"separator":"[\\s\\-]+","pipeline":["stopWordFilter"],"fields":{"title":{"boost":1000.0},"text":{"boost":1.0},"tags":{"boost":1000000.0}}},"docs":[{"location":"index.html","title":"Overview","text":"<p>GenJAX: a probabilistic programming library designed from the ground up to scale Bayesian modeling and inference into high performance settings. (1)</p> <ol> <li> <p>Here, high performance means massively parallel, either cores or devices.</p> <p>For those whom this overview page may be irrelevant: the value proposition is about putting expressive models and customizable Bayesian inference on GPUs, TPUs, etc - without sacrificing abstraction or modularity.</p> </li> </ol> <p>Gen is a multi-paradigm (generative, differentiable, incremental) system for probabilistic programming. GenJAX is an implementation of Gen on top of JAX (2) - exposing the ability to programmatically construct and manipulate generative functions (1) (computational objects which represent probability measures over structured sample spaces), with compilation to native devices, accelerators, and other parallel fabrics. </p> <ol> <li> <p>By design, generative functions expose a concise interface for expressing approximate and differentiable inference algorithms. </p> <p>The set of generative functions is extensible! You can implement your own - allowing advanced users to performance optimize their critical modeling/inference code paths.</p> <p>You can (and we, at the MIT Probabilistic Computing Project, do!) use these objects for machine learning - including robotics, natural language processing, reasoning about agents, and modelling / creating systems which exhibit human-like reasoning.</p> <p>A precise mathematical formulation of generative functions is given in Marco Cusumano-Towner's PhD thesis.</p> </li> <li> <p>If the usage of JAX is not a dead giveaway, GenJAX is written in Python.</p> </li> </ol> Model codeInference code <p><p> Defining a beta-bernoulli process model as a generative function in GenJAX. </p></p> <pre><code>@genjax.gen\ndef model():\np = beta(0, 1) @ \"p\"\nv = bernoulli(p) @ \"v\"\nreturn v\n</code></pre> <p><p> This works for any generative function, not just the beta-bernoulli model. </p></p> <pre><code># Sampling importance resampling.\ndef sir(key: PRNGKey, gen_fn: GenerativeFunction, model_args: Tuple,\nobs: ChoiceMap, n_samples: Int):\nkey, sub_keys = genjax.slash(key, n_samples) # split keys\n_, (lws, trs) = jax.vmap(gen_fn.importance, in_axes=(0, None, None))(\nsub_keys,\nobs,\nargs,\n)\nlog_total_weight = jax.scipy.special.logsumexp(lws)\nlog_normalized_weights = lws - log_total_weight\nlog_ml_estimate = log_total_weight - jnp.log(self.num_particles)\nreturn key, (trs, log_normalized_weights, log_ml_estimate)\n</code></pre>"},{"location":"index.html#what-sort-of-things-do-you-use-genjax-for","title":"What sort of things do you use GenJAX for?","text":"Real time object tracking <p>Real time tracking of objects in 3D using probabilistic rendering. (Left) Ground truth, (center) depth mask, (right) inference overlaid on ground truth.</p> <p><p> </p></p>"},{"location":"index.html#why-gen","title":"Why Gen?","text":"<p>GenJAX is a Gen implementation. If you're considering using GenJAX - it's worth starting by understanding what problems Gen purports to solve.</p>"},{"location":"index.html#the-evolution-of-probabilistic-programming-languages","title":"The evolution of probabilistic programming languages","text":"<p>Probabilistic modeling and inference is hard: understanding a domain well enough to construct a probabilistic model in the Bayesian paradigm is challenging, and that's half the battle - the other half is designing effective inference algorithms to probe the implications of the model (1).</p> <ol> <li> <p>Some probabilistic programming languages restrict the set of allowable models, providing (in return) efficient (often, exact) inference. </p> <p>Gen considers a wide class of models - include Bayesian nonparametrics, open-universe models, and models over rich structures (like programs!) - which don't natively support efficient exact inference.</p> </li> </ol> <p>Probabilistic have historically considered the following design loop.</p> <pre><code>graph LR\n  A[Design model.] --&gt; B[Implement inference by hand.];\n  B --&gt; C[Model + inference okay?];\n  C --&gt; D[Happy.];\n  C --&gt; A;</code></pre> <p>The first generation (1) of probabilistic programming systems introduced inference engines which could operate abstractly over many different models, without requiring the programmer to return and tweak their inference code. The utopia envisioned by these systems is shown below.</p> <ol> <li> <p>Here, the definition of \"first generation\" includes systems like JAGS, BUGS, BLOG, IBAL, Church, Infer.NET, Figaro, Stan, amongst others.</p> <p>But more precisely, many systems preceded the DARPA PPAML project - which gave rise to several novel systems, including the predecessors of Gen.</p> </li> </ol> <pre><code>graph LR\n  A[Design model.] --&gt; D[Model + inference okay?];\n  B[Inference engine.] ---&gt; D;\n  D --&gt; E[Happy.];\n  D ---&gt; A;</code></pre> <p>The problem with this utopia is that we often need to customize our inference algorithms (1) to achieve maximum performance, with respect to accuracy as well as runtime (2). First generation systems were not designed with this in mind.</p> <ol> <li>Here, programmable inference denotes using a custom proposal distribution in importance sampling, or a custom variational family for variational inference, or even a custom kernel in Markov chain Monte Carlo.</li> <li>Composition of inference programs can also be highly desirable when performing inference in complex models, or designing a probabilistic application from several modeling and inference components. The first examples of universal inference engines ignored this design problem.</li> </ol>"},{"location":"index.html#programmable-inference","title":"Programmable inference","text":"<p>A worthy design goal is to allow users to customize when required, while retaining the rapid model/inference iteration properties explored by first generation systems.</p> <p>Gen addresses this goal by introducing a separation between modeling and inference code: the generative function interface.</p> <p> </p> <p>The interface provides an abstraction layer that inference algorithms can call to compute the necessary (and hard to get right!) math (1). Probabilistic application developers can also extend the interface to new modeling languages - and immediately gain access to advanced inference procedures.</p> <ol> <li> <p>Examples of hard-to-get-right math: importance weights, accept reject ratios, and gradient estimators. </p> <p>For simple models and inference, one might painlessly derive these quantities. As soon as the model/inference gets complicated, however, you might find yourself thanking the interface.</p> </li> </ol>"},{"location":"index.html#whose-using-gen","title":"Whose using Gen?","text":"<p>Gen supports a growing list of users, with collaboration across academic research labs and industry affiliates.</p> <p> </p> <p>We're looking to expand our user base! If you're interested, please contact us to get involved.</p>"},{"location":"genjax/diff_jl.html","title":"Diffing against Gen.jl","text":"<p><code>GenJAX</code> is inherits concepts from Gen and algorithm reference implementations from <code>Gen.jl</code> - there are a few necessary design deviations between <code>GenJAX</code> and <code>Gen.jl</code> that stem from JAX's underlying array programming model. In this section, we describe several of these differences and try to highlight workarounds or discuss the reason for the discrepancy.</p>"},{"location":"genjax/diff_jl.html#turing-universality","title":"Turing universality","text":"<p><code>Gen.jl</code> is Turing universal - it can encode any computable distribution, including those expressed by forms of unbounded recursion.</p> <p>It is a bit ambiguous whether or not <code>GenJAX</code> falls in this category: JAX does not feature mechanisms for dynamic shape allocations, but it does feature mechanisms for unbounded recursion.</p> <p>The former provides a technical barrier to implementing Gen's trace machinery. While JAX allows for unbounded recursion, to support Gen's interfaces we also need the ability to dynamically allocate choice data. This requirement is currently at tension with XLA's requirements of knowing the static shape of everything.</p> <p>However, <code>GenJAX</code> supports generative function combinators with bounded recursion / unfold chain length. Ahead of time, these combinators can be directed to pre-allocate arrays with enough size to handle recursion/looping within the bounds that the programmer sets. If these bounds are exceeded, a Python runtime error will be thrown (both on and off JAX device).</p> <p>In practice, this means that some performance engineering (space vs. expressivity) is required of the programmer. It's certainly feasible to express bounded recursive computations which terminate with probability 1 - but you'll need to ahead of time allocate space for it.</p>"},{"location":"genjax/diff_jl.html#mutation","title":"Mutation","text":"<p>Just like JAX, GenJAX disallows mutation - expressing a mutation to an array must be done through special interfaces, and those interfaces return full copies. There are special circumstances where these interfaces will be performed in place.</p>"},{"location":"genjax/diff_jl.html#to-jit-or-not-to-jit","title":"To JIT or not to JIT","text":"<p><code>Gen.jl</code> is written in Julia, which automatically JITs everything. <code>GenJAX</code>, by virtue of being constructed on top of JAX, allows us to JIT JAX compatible code - but the JIT process is user directed. Thus, the idioms that are used to express and optimize inference code are necessarily different compared to <code>Gen.jl</code>. In the inference standard library, you'll typically find algorithms implemented as dataclasses which inherit (and implement) the <code>jax.Pytree</code> interfaces. Implementing these interfaces allow usage of inference dataclasses and methods in jittable code - and, as a bonus, allow us to be specific about trace vs. runtime known values.</p> <p>In general, it's productive to enclose as much of a computation as possible in a <code>jax.jit</code> block. This can sometimes lead to long trace times. If trace times are ballooning, a common source is explicit for-loops (with known bounds, else JAX will complain). In these cases, you might look at Advice on speeding up compilation time. We've taken care to optimize (by e.g. using XLA primitives) the code which we expose from GenJAX - but if you find something out of the ordinary, file an issue!</p>"},{"location":"genjax/language_aperitifs.html","title":"Language ap\u00e9ritifs","text":"<p>The implementation of GenJAX adhers to commonly accepted JAX idioms (1) and modern functional programming patterns (2).</p> <ol> <li>One example: everything is a Pytree. Implies another: everything is JAX traceable by default.</li> <li>Modern here meaning patterns concerning the composition of effectful computations via effect handling abstractions.</li> </ol> <p>GenJAX consists of a set of languages based around transforming pure functions to apply semantic transformations. In this page, we'll provide a taste of some of these languages.</p>"},{"location":"genjax/language_aperitifs.html#the-builtin-language","title":"The builtin language","text":"<p>GenJAX provides a builtin language which supports a <code>trace</code> primitive and the ability to invoke other generative functions as callees:</p> <pre><code>@genjax.gen\ndef submodel():\nx = trace(\"x\", normal)(0.0, 1.0) # explicit\nreturn x\n@genjax.gen\ndef model():\nx = submodel() @ \"sub\" # sugared\nreturn x\n</code></pre> <p>The <code>trace</code> call is a JAX primitive which is given semantics by transformations which implement the semantics of inference interfaces described in Generative functions.</p> <p>Addresses (here, <code>\"x\"</code> and <code>\"sub\"</code>) are important - addressed random choices within <code>trace</code> allow us to structure the address hierarchy for the measure over choice maps which generative functions in this language define.</p> <p>Because convenient idioms for working with addresses is so important in Gen, the generative functions from the builtin language also support a form of \"splatting\" addresses into a caller.</p> <pre><code>@genjax.gen\ndef model():\nx = submodel.inline()\nreturn x\n</code></pre> <p>Invoking the <code>submodel</code> via the <code>inline</code> interface here means that the addresses in <code>submodel</code> are flattened into the address level for the <code>model</code>. If there's overlap, that's a problem! But GenJAX will yell at you for that.</p>"},{"location":"genjax/language_aperitifs.html#structured-control-flow-with-combinators","title":"Structured control flow with combinators","text":"<p>The base modeling language is the <code>BuiltinGenerativeFunction</code> language shown above. The builtin language is based on pure functions, with the interface semantics implemented using program transformations. But we'd also like to take advantage of structured control flow in our generative computations. </p> <p>Users gain access to structured control flow via combinators, other generative function mini-languages which implement the interfaces in control flow compatible ways.</p> <pre><code>@functools.partial(genjax.Map, in_axes=(0, 0))\n@genjax.gen\ndef kernel(x, y):\nz = normal(x + y, 1.0) @ \"z\"\nreturn z\n</code></pre> <p>This defines a <code>MapCombinator</code> generative function - a generative function whose interfaces take care of applying <code>vmap</code> in the appropriate ways (1).</p> <ol> <li>Read: compatible with JIT, gradients, and incremental computation.</li> </ol> <p><code>MapCombinator</code> has a vectorial friend named <code>UnfoldCombinator</code> which implements a <code>scan</code>-like pattern of generative computation.</p> <pre><code>@functools.partial(genjax.Unfold, max_length = 10)\n@genjax.gen\ndef kernel(prev, static_args):\nsigma, = static_args\nnew = normal(prev, sigma) @ \"z\"\nreturn new\n</code></pre> <p><code>UnfoldCombinator</code> allows the expression of general state space models - modeled as a generative function which supports a dependent-for (1) control flow pattern.</p> <ol> <li>Dependent-for means that each iteration may depend on the output from the previous iteration. Think of <code>jax.lax.scan</code> here.</li> </ol> <p><code>UnfoldCombinator</code> allows uncertainty over the length of the chain:</p> <pre><code>@genjax.gen\ndef top_model(p):\nlength = truncated_geometric(10, p) @ \"l\"\ninitial_state = normal(0.0, 1.0) @ \"init\"\nsigma = normal(0.0, 1.0) @ \"sigma\"\n(v, xs) = scanner(length, initial_state, sigma)\nreturn v\n</code></pre> <p>Here, <code>length</code> is drawn from a truncated geometric distribution, and determines the index range of the chain which participates in the generative computation.</p> <p>Of course, combinators are composable.</p> <pre><code>@functools.partial(genjax.Map, in_axes = (0, ))\n@genjax.gen\ndef top_model(p):\nlength = truncated_geometric(10, p) @ \"l\"\ninitial_state = normal(0.0, 1.0) @ \"init\"\nsigma = normal(0.0, 1.0) @ \"sigma\"\n(v, xs) = scanner(length, initial_state, sigma)\nreturn v\n</code></pre> <p>Now we're describing a broadcastable generative function whose internal choices include a chain-like generative structure with dynamic truncation using padding. And we could go on!</p>"},{"location":"genjax/language_aperitifs.html#approximation-of-marginal-densities","title":"Approximation of marginal densities","text":"<p>GenJAX also features an implementation of a framework for pseudomarginalization and approximate normalization called <code>Prox</code>. <code>Prox</code> allows construction of approximate distributions, distributions whose sample and density interfaces are replaced with unbiased estimators.</p>"},{"location":"genjax/language_aperitifs.html#automatic-differentiation-of-expected-values","title":"Automatic differentiation of expected values","text":"<p>JAX is a state-of-the-art framework for AD. Not to be outdone, GenJAX features a state-of-the-art framework for constructing unbiased gradient estimators for loss functions defined as expected values (1).</p> <ol> <li>Lew &amp; Huot et al, 2022, ADEV: Sound Automatic Differentiation of Expected Values of Probabilistic Programs</li> </ol> \\[ L(\\theta) = E_{v \\sim P(\\cdot \\ ; \\ \\theta)}[f(v, \\theta)] \\]"},{"location":"genjax/notebooks.html","title":"Modeling &amp; inference notebooks","text":"<p>Link to the notebook repository</p> <p>This section contains a link to a (statically hosted) series of tutorial notebooks designed to guide usage of GenJAX. These notebooks are executed and rendered with quarto, and are kept up to date with the repository along with the documentation.</p> <p>The notebook repository can be found here.</p>"},{"location":"genjax/concepts/index.html","title":"A general-purpose PPL with programmable inference","text":""},{"location":"genjax/concepts/generative_functions.html","title":"Generative functions","text":"<p>Gen is all about generative functions: computational objects which support an interface that helps automate the tricky math involved in programming Bayesian inference algorithms. In this section, we'll unpack the generative function interface and explain the mathematics behind generative functions (1).</p> <ol> <li>For a deeper dive, enjoy Marco Cusumano-Towner's PhD thesis.</li> </ol>"},{"location":"genjax/library/core.html","title":"Core","text":"<p>This module provides the core functionality and JAX compatibility layer which <code>GenJAX</code> generative function and inference modules are built on top of. It contains (truncated, and in no particular order):</p> <ul> <li> <p>Core Gen associated data types for generative functions.</p> </li> <li> <p>Utility functionality for automatically registering class definitions as valid <code>Pytree</code> method implementors (guaranteeing <code>flatten</code>/<code>unflatten</code> compatibility across JAX transform boundaries). For more information, see Pytrees.</p> </li> <li> <p>Staging functionality that allows lifting of pure, numerical Python programs to <code>ClosedJaxpr</code> instances.</p> </li> <li> <p>Transformation interpreters: interpreter-based transformations on which operate on <code>ClosedJaxpr</code> instances. Interpreters are all written in initial style - they operate on <code>ClosedJaxpr</code> instances, and don't implement their own custom <code>jax.Tracer</code> types - but they are JAX compatible, implying that they can be staged out for zero runtime cost.</p> </li> <li> <p>Masking functionality which allows active/inactive flagging of data - useful when branching patterns of computation require uncertainty in whether or not data is active with respect to a generative computation.</p> </li> </ul>"},{"location":"genjax/library/core.html#generative-functions","title":"Generative functions","text":"<p>The main computational object of Gen is the generative function. These objects support a method and associated type interface which allows inference layers to abstract over the interface implementation.</p> <p>Below, we document the base abstract class. Concrete generative function languages are described in their own documentation module.</p>"},{"location":"genjax/library/core.html#genjax.core.GenerativeFunction","title":"<code>genjax.core.GenerativeFunction</code>  <code>dataclass</code>","text":"<p>         Bases: <code>Pytree</code></p> <p>Abstract class which provides an inheritance base for user-defined implementations of the generative function interface methods e.g. the <code>BuiltinGenerativeFunction</code> and <code>Distribution</code> languages both implement a class inheritor of <code>GenerativeFunction</code>.</p> <p>Any implementation will interact with the JAX tracing machinery, however, so there are specific API requirements above the requirements enforced in other languages (unlike Gen in Julia, for example).</p> <p>The user must match the interface signatures of the native JAX implementation. This is not statically checked - but failure to do so will lead to unintended behavior or errors.</p> <p>To support argument and choice gradients via JAX, the user must provide a differentiable <code>importance</code> implementation.</p> Source code in <code>src/genjax/_src/core/datatypes/generative.py</code> <pre><code>@dataclasses.dataclass\nclass GenerativeFunction(Pytree):\n\"\"\"Abstract class which provides an inheritance base for user-defined\n    implementations of the generative function interface methods e.g. the\n    `BuiltinGenerativeFunction` and `Distribution` languages both implement a\n    class inheritor of `GenerativeFunction`.\n    Any implementation will interact with the JAX tracing machinery,\n    however, so there are specific API requirements above the requirements\n    enforced in other languages (unlike Gen in Julia, for example).\n    The user *must* match the interface signatures of the native JAX\n    implementation. This is not statically checked - but failure to do so\n    will lead to unintended behavior or errors.\n    To support argument and choice gradients via JAX, the user must\n    provide a differentiable `importance` implementation.\n    \"\"\"\n# This is used to support tracing -- the user is not required to provide\n# a PRNGKey, because the value of the key is not important, only\n# the fact that the value has type PRNGKey.\ndef __abstract_call__(self, *args) -&gt; Tuple[PRNGKey, Any]:\nkey = jax.random.PRNGKey(0)\n_, tr = self.simulate(key, args)\nretval = tr.get_retval()\nreturn retval\ndef get_trace_type(self, *args, **kwargs) -&gt; TraceType:\nshape = kwargs.get(\"shape\", ())\nreturn Bottom(shape)\n@abc.abstractmethod\ndef simulate(\nself,\nkey: PRNGKey,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Trace]:\npass\n@abc.abstractmethod\ndef importance(\nself,\nkey: PRNGKey,\nchm: ChoiceMap,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[FloatArray, Trace]]:\npass\n@abc.abstractmethod\ndef update(\nself,\nkey: PRNGKey,\noriginal: Trace,\nnew: ChoiceMap,\ndiffs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[Any, FloatArray, Trace, ChoiceMap]]:\npass\n@abc.abstractmethod\ndef assess(\nself,\nkey: PRNGKey,\nevaluation_point: ChoiceMap,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[Any, FloatArray]]:\npass\ndef unzip(\nself,\nkey: PRNGKey,\nfixed: ChoiceMap,\n) -&gt; Tuple[\nPRNGKey,\nCallable[[ChoiceMap, Tuple], FloatArray],\nCallable[[ChoiceMap, Tuple], Any],\n]:\nkey, sub_key = jax.random.split(key)\ndef score(differentiable: Tuple, nondifferentiable: Tuple) -&gt; FloatArray:\nprovided, args = tree_zipper(differentiable, nondifferentiable)\nmerged = fixed.merge(provided)\n_, (_, score) = self.assess(sub_key, merged, args)\nreturn score\ndef retval(differentiable: Tuple, nondifferentiable: Tuple) -&gt; Any:\nprovided, args = tree_zipper(differentiable, nondifferentiable)\nmerged = fixed.merge(provided)\n_, (retval, _) = self.assess(sub_key, merged, args)\nreturn retval\nreturn key, score, retval\n# A higher-level gradient API - it relies upon `unzip`,\n# but provides convenient access to first-order gradients.\ndef choice_grad(self, key, trace, selection):\nfixed = selection.complement().filter(trace.strip())\nevaluation_point = selection.filter(trace.strip())\nkey, scorer, _ = self.unzip(key, fixed)\ngrad, nograd = tree_grad_split(\n(evaluation_point, trace.get_args()),\n)\nchoice_gradient_tree, _ = jax.grad(scorer)(grad, nograd)\nreturn key, choice_gradient_tree\n###################\n# ADEV and fusion #\n###################\ndef adev_simulate(self, key: PRNGKey, args: Tuple) -&gt; Tuple[PRNGKey, Any]:\n\"\"\"An opt-in method which expresses forward sampling from the\n        generative function in terms of primitives which are compatible with\n        ADEV's language.\"\"\"\nraise NotImplementedError\ndef prepare_fuse(self, key: PRNGKey, args: Tuple):\n\"\"\"Convert a generative function to a canonical form with ADEV\n        primitives for proposal fusion.\"\"\"\nraise NotImplementedError\ndef fuse(self, _: \"GenerativeFunction\"):\n\"\"\"Fuse a generative function and a proposal to produce a probabilistic\n        computation that returns an ELBO estimate.\"\"\"\nraise NotImplementedError\n</code></pre>"},{"location":"genjax/library/core.html#genjax._src.core.datatypes.generative.GenerativeFunction.simulate","title":"<code>simulate(key, args)</code>  <code>abstractmethod</code>","text":"Source code in <code>src/genjax/_src/core/datatypes/generative.py</code> <pre><code>@abc.abstractmethod\ndef simulate(\nself,\nkey: PRNGKey,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Trace]:\npass\n</code></pre>"},{"location":"genjax/library/core.html#genjax._src.core.datatypes.generative.GenerativeFunction.importance","title":"<code>importance(key, chm, args)</code>  <code>abstractmethod</code>","text":"Source code in <code>src/genjax/_src/core/datatypes/generative.py</code> <pre><code>@abc.abstractmethod\ndef importance(\nself,\nkey: PRNGKey,\nchm: ChoiceMap,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[FloatArray, Trace]]:\npass\n</code></pre>"},{"location":"genjax/library/core.html#genjax._src.core.datatypes.generative.GenerativeFunction.update","title":"<code>update(key, original, new, diffs)</code>  <code>abstractmethod</code>","text":"Source code in <code>src/genjax/_src/core/datatypes/generative.py</code> <pre><code>@abc.abstractmethod\ndef update(\nself,\nkey: PRNGKey,\noriginal: Trace,\nnew: ChoiceMap,\ndiffs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[Any, FloatArray, Trace, ChoiceMap]]:\npass\n</code></pre>"},{"location":"genjax/library/core.html#genjax._src.core.datatypes.generative.GenerativeFunction.assess","title":"<code>assess(key, evaluation_point, args)</code>  <code>abstractmethod</code>","text":"Source code in <code>src/genjax/_src/core/datatypes/generative.py</code> <pre><code>@abc.abstractmethod\ndef assess(\nself,\nkey: PRNGKey,\nevaluation_point: ChoiceMap,\nargs: Tuple,\n) -&gt; Tuple[PRNGKey, Tuple[Any, FloatArray]]:\npass\n</code></pre>"},{"location":"genjax/library/core.html#interpreters","title":"Interpreters","text":"<p>JAX supports transformations of pure, numerical Python programs by staging out interpreters which evaluate <code>Jaxpr</code> representations of programs.</p> <p>The <code>Core</code> module features interpreter infrastructure, and common transforms designed to facilitate certain types of transformations.</p>"},{"location":"genjax/library/core.html#contextual-interpreter","title":"Contextual interpreter","text":"<p>A common type of interpreter involves overloading desired primitives with context-specific behavior by inheriting from <code>Trace</code> and define the correct methods to process the primitives.</p> <p>In this module, we provide an interpreter which mixes initial style (e.g. the Python program is immediately staged, and then an interpreter walks the <code>Jaxpr</code> representation) with custom <code>Trace</code> and <code>Tracer</code> overloads. </p> <p>This pattern supports a wide range of program transformations, and allows parametrization over the inner interpreter (e.g. forward evaluation, or CPS).</p>"},{"location":"genjax/library/core.html#genjax._src.core.interpreters.context","title":"<code>genjax._src.core.interpreters.context</code>","text":"<p>This module contains a transformation infrastructure based on interpreters with stateful contexts and custom primitive handling lookups.</p>"},{"location":"genjax/library/core.html#genjax._src.core.interpreters.context.ContextualTracer","title":"<code>ContextualTracer</code>","text":"<p>         Bases: <code>jc.Tracer</code></p> <p>A <code>ContextualTracer</code> encapsulates a single value.</p> Source code in <code>src/genjax/_src/core/interpreters/context.py</code> <pre><code>class ContextualTracer(jc.Tracer):\n\"\"\"A `ContextualTracer` encapsulates a single value.\"\"\"\ndef __init__(self, trace: \"ContextualTrace\", val: Value):\nself._trace = trace\nself.val = val\n@property\ndef aval(self):\nreturn abstract_arrays.raise_to_shaped(jc.get_aval(self.val))\ndef full_lower(self):\nreturn self\n</code></pre>"},{"location":"genjax/library/core.html#genjax._src.core.interpreters.context.ContextualTrace","title":"<code>ContextualTrace</code>","text":"<p>         Bases: <code>jc.Trace</code></p> <p>An evaluating trace that dispatches to a dynamic context.</p> Source code in <code>src/genjax/_src/core/interpreters/context.py</code> <pre><code>class ContextualTrace(jc.Trace):\n\"\"\"An evaluating trace that dispatches to a dynamic context.\"\"\"\ndef pure(self, val: Value) -&gt; ContextualTracer:\nreturn ContextualTracer(self, val)\ndef sublift(self, tracer: ContextualTracer) -&gt; ContextualTracer:\nreturn self.pure(tracer.val)\ndef lift(self, val: Value) -&gt; ContextualTracer:\nreturn self.pure(val)\ndef process_primitive(\nself,\nprimitive: jc.Primitive,\ntracers: List[ContextualTracer],\nparams: Dict[str, Any],\n) -&gt; Union[ContextualTracer, List[ContextualTracer]]:\ncontext = staging.get_dynamic_context(self)\ncustom_rule = context.get_custom_rule(primitive)\nif custom_rule:\nreturn custom_rule(self, *tracers, **params)\nreturn self.default_process_primitive(primitive, tracers, params)\ndef default_process_primitive(\nself,\nprimitive: jc.Primitive,\ntracers: List[ContextualTracer],\nparams: Dict[str, Any],\n) -&gt; Union[ContextualTracer, List[ContextualTracer]]:\ncontext = staging.get_dynamic_context(self)\nvals = [v.val for v in tracers]\nif context.can_process(primitive):\noutvals = context.process_primitive(primitive, *vals, **params)\nreturn jax_util.safe_map(self.pure, outvals)\noutvals = primitive.bind(*vals, **params)\nif not primitive.multiple_results:\noutvals = [outvals]\nout_tracers = jax_util.safe_map(self.full_raise, outvals)\nif primitive.multiple_results:\nreturn out_tracers\nreturn out_tracers[0]\ndef process_call(\nself,\ncall_primitive: jc.Primitive,\nf: Any,\ntracers: List[ContextualTracer],\nparams: Dict[str, Any],\n):\ncontext = staging.get_dynamic_context(self)\nreturn context.process_higher_order_primitive(\nself, call_primitive, f, tracers, params, False\n)\ndef post_process_call(self, call_primitive, out_tracers, params):\nvals = tuple(t.val for t in out_tracers)\nmaster = self.main\ndef todo(x):\ntrace = ContextualTrace(master, jc.cur_sublevel())\nreturn jax_util.safe_map(functools.partial(ContextualTracer, trace), x)\nreturn vals, todo\ndef process_map(\nself,\ncall_primitive: jc.Primitive,\nf: Any,\ntracers: List[ContextualTracer],\nparams: Dict[str, Any],\n):\ncontext = staging.get_dynamic_context(self)\nreturn context.process_higher_order_primitive(\nself, call_primitive, f, tracers, params, True\n)\npost_process_map = post_process_call\ndef process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):\ncontext = staging.get_dynamic_context(self)\nreturn context.process_custom_jvp_call(\nself, primitive, fun, jvp, tracers, symbolic_zeros=symbolic_zeros\n)\ndef post_process_custom_jvp_call(self, out_tracers, jvp_was_run):\ncontext = staging.get_dynamic_context(self)\nreturn context.post_process_custom_jvp_call(self, out_tracers, jvp_was_run)\ndef process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):\ncontext = staging.get_dynamic_context(self)\nreturn context.process_custom_vjp_call(\nself, primitive, fun, fwd, bwd, tracers, out_trees\n)\ndef post_process_custom_vjp_call(self, out_tracers, params):\ncontext = staging.get_dynamic_context(self)\nreturn context.post_process_custom_vjp_call(self, out_tracers, params)\ndef post_process_custom_vjp_call_fwd(self, out_tracers, out_trees):\ncontext = staging.get_dynamic_context(self)\nreturn context.post_process_custom_vjp_call_fwd(self, out_tracers, out_trees)\n</code></pre>"},{"location":"genjax/library/core.html#harvest-via-contextual-interpreter","title":"<code>harvest</code> via contextual interpreter","text":""},{"location":"genjax/library/inference.html","title":"Inference","text":""},{"location":"genjax/library/generative_functions/index.html","title":"Index","text":"<p>This module contains several standard generative function classes useful for structuring probabilistic programs.</p>"},{"location":"genjax/library/generative_functions/builtin.html","title":"Builtin language","text":"<p>This module provides a function-like modeling language. The generative function interfaces are implemented for objects in this language using transformations by JAX interpreters.</p> <p>The language also exposes a set of JAX primitives which allow hierarchical construction of generative programs. These programs can utilize other generative functions inside of a new JAX primitive (<code>trace</code>) to create hierarchical patterns of generative computation.</p> <p>The <code>Builtin</code> language is a common foundation for constructing models. It exposes a DSL based on JAX primitives and transformations which allows the programmer to construct generative functions out of Python functions. Below, we illustrate a simple example:</p> <pre><code>from genjax import beta \nfrom genjax import bernoulli \nfrom genjax import uniform \nfrom genjax import gen\n@genjax.gen\ndef beta_bernoulli_process(u):\np = beta(0, u) @ \"p\"\nv = bernoulli(p) @ \"v\"\nreturn v\n@genjax.gen\ndef joint():\nu = uniform() @ \"u\"\nv = beta_bernoulli_process(u) @ \"bbp\"\nreturn v\n</code></pre>"},{"location":"genjax/library/generative_functions/builtin.html#control-flow-within-the-dsl","title":"Control flow within the DSL","text":""}]}