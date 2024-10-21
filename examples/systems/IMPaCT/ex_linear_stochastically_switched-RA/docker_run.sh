#! /bin/bash

docker run --mount type=bind,src=$PWD/systems/IMPaCT/ex_linear_stochastically_switched-RA,dst=/app/examples/ex_linear_stochastically_switched-RA ghcr.io/kiguli/impact:main examples/ex_linear_stochastically_switched-RA/run_benchmark.sh
