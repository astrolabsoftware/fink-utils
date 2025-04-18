#!/bin/bash
# Copyright 2022 Astrolab Software
# Author: Le Montagner Roman
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
## Script to launch the python test suite and measure the coverage.
set -e

SINFO="\xF0\x9F\x9B\x88"
SERROR="\xE2\x9D\x8C"
SSTOP="\xF0\x9F\x9B\x91"
SSTEP="\xF0\x9F\x96\xA7"
SDONE="\xE2\x9C\x85"

message_help="""
Run the test suite of the modules\n\n
Usage:\n
    \t./run_tests.sh [--single_module]\n\n

Note you need Spark 3.1.3+ installed to fully test the modules.
"""

export ROOTPATH=`pwd`

while [ "$#" -gt 0 ]; do
  case "$1" in
    --single_module)
      SINGLE_MODULE_PATH=$2
      shift 2
      ;;
    -h)
        echo -e $message_help
        exit
        ;;
  esac
done

export PYTHONPATH="${SPARK_HOME}/python/test_coverage:$PYTHONPATH"
export COVERAGE_PROCESS_START=".coveragerc"

# single module testing
if [[ -n "${SINGLE_MODULE_PATH}" ]]; then
  coverage run \
   --source=${ROOTPATH} \
   --rcfile ${ROOTPATH}/.coveragerc ${SINGLE_MODULE_PATH}

  # Combine individual reports in one
  coverage combine

  unset COVERAGE_PROCESS_START

  coverage report -m
  coverage html

  exit 0

fi

for filename in fink_utils/sso/*.py
do
    echo $filename
    coverage run \
    --source=${ROOTPATH} \
    --rcfile .coveragerc \
    $filename
done

for filename in fink_utils/photometry/*.py
do
    echo $filename
    coverage run \
    --source=${ROOTPATH} \
    --rcfile .coveragerc \
    $filename
done

for filename in fink_utils/logging/*.py
do
    echo $filename
    coverage run \
    --source=${ROOTPATH} \
    --rcfile .coveragerc \
    $filename
done

for filename in fink_utils/cutouts/*.py
do
    echo $filename
    coverage run \
    --source=${ROOTPATH} \
    --rcfile .coveragerc \
    $filename
done

echo fink_utils/slack_bot/bot_test.py
coverage run --source=${ROOTPATH} --rcfile .coveragerc fink_utils/slack_bot/bot_test.py

coverage combine

unset FINK_PACKAGES
unset FINK_JARS

coverage report -m
coverage html
