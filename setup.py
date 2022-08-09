# Copyright 2022 MIT ProbComp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

from setuptools import setup

# Specify the requirements.
requirements = {
    "genjax": [
        "jax==0.3.13",
    ],
}
requirements["all"] = [r for v in requirements.values() for r in v]

# Determine the version (hardcoded).
dirname = os.path.dirname(os.path.realpath(__file__))
vre = re.compile('__version__ = "(.*?)"')
m = open(os.path.join(dirname, "genjax", "__init__.py")).read()
__version__ = vre.findall(m)[0]

setup(
    name="genjax",
    version=__version__,
    description="Gen ⊗ JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/probcomp/genjax",
    license="Apache-2.0",
    maintainer="McCoy R. Becker",
    maintainer_email="mccoyb@mit.edu",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages=[
        "genjax",
    ],
    package_dir={
        "genjax": "genjax",
    },
    install_requires=requirements["genjax"],
    extras_require=requirements,
    python_requires=">=3.8",
)
