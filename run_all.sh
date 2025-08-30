#!/usr/bin/env bash
set -euo pipefail

make data
make install
make profile
make baseline
