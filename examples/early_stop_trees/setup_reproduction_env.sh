#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
submodule="$repo_root/treeple/_lib/sklearn_fork"
patch_file="$repo_root/examples/early_stop_trees/patches/sklearn_splitter_kwargs.patch"
requirements="$repo_root/examples/early_stop_trees/requirements-reproduction.txt"
python_bin=${PYTHON:-python3}

git -C "$repo_root" submodule update --init -- treeple/_lib/sklearn_fork

if grep -q '        \*\*kwargs' "$submodule/sklearn/tree/_splitter.pyx"; then
    printf '%s\n' 'scikit-learn splitter patch already applied.'
elif git -C "$submodule" apply --check "$patch_file"; then
    git -C "$submodule" apply "$patch_file"
    printf '%s\n' 'Applied the early-stop splitter compatibility patch.'
else
    printf '%s\n' 'Unable to apply the pinned splitter patch.' >&2
    exit 1
fi

ln -sfn sklearn_fork/sklearn "$repo_root/treeple/_lib/sklearn"

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    printf '%s\n' 'Dependency source prepared; package installation skipped.'
    exit 0
fi

"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install -r "$requirements"
"$python_bin" -m pip install "$repo_root"

printf '%s\n' 'Reproduction environment installed.'
