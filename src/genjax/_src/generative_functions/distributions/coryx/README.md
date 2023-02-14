# `coryx` (distribution transformation DSL)

This module supports a distribution DSL similar to [Gen.jl][gen_dist]. The design of the DSL is heavily inspired by the two [Oryx][oryx] core transformations `inverse` and `ildj`.

We utilize these core transformations to support a generative function language which allows sampling the values of random variables from other `Distribution` generative functions, and transforming them with a function `f` which is compatible with the `ildj` transformation.

In the future (pending more information about Oryx's development model), this DSL may rely on Oryx directly for its functionality - for now, we've forked the code here.

## Code copyright

Unless otherwise noted, the following copyright attribution notice applies to all code in this directory.

> Copyright 2022 The oryx Authors and the MIT Probabilistic Computing Project.
>
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.

[gen_dist]: https://www.gen.dev/docs/stable/ref/distributions/#dist_dsl-1
[oryx]: https://github.com/jax-ml/oryx
