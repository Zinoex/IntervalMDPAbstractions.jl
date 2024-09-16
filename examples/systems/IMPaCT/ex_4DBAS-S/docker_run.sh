#! /bin/bash

docker run --mount type=bind,src=$PWD/systems/IMPaCT/ex_4DBAS-S,dst=/app/examples/ex_4DBAS-S ghcr.io/kiguli/impact:main examples/ex_4DBAS-S/run_benchmark.sh