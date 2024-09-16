#! /bin/bash

docker run --mount type=bind,src=$PWD/systems/IMPaCT/ex_van_der_pol-R,dst=/app/examples/ex_van_der_pol-R ghcr.io/kiguli/impact:main examples/ex_van_der_pol-R/run_benchmark.sh