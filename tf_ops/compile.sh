#!/bin/bash
echo "compiling cal ivt..."
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}/calivt/
sh tf_calivt_compile.sh