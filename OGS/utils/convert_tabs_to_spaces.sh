#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=${1:-"$(cd -- "$script_dir/../../.." && pwd)"}

find "$project_root" \
  -type f \
  -name '*.py' \
  -not -path '*/.git/*' \
  -not -path '*/.miniconda3/*' \
  -print0 |
  while IFS= read -r -d '' python_file; do
    perl -pi -e 's/^([ \t]*)\t/$1  / while /^[ \t]*\t/' "$python_file"
  done
