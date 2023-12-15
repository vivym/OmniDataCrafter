#!/bin/bash
set -e
set -u

SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

docker run -it --rm --ipc=host --network=host --gpus=all -v $PWD:/workspace/OmniDataCrafter odc-dev
