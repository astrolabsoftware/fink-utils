# Copyright 2022 AstroLab Software
# Author: Julien Peloton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import setuptools
from fink_utils import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fink-utils",
    version=__version__,
    author="JulienPeloton",
    author_email="peloton@lal.in2p3.fr",
    description=" Collection of useful functionalities used across the Fink ecosystem.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/astrolabsoftware/fink-utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    project_urls={"Source": "https://github.com/astrolabsoftware/fink-utils"},
    package_data={"fink_utils": ["xmatch/otypes.txt"]},
)
