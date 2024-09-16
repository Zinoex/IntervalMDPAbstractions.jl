#! /bin/bash

docker run --mount type=bind,src=$PWD/../ex_7DBAS-S,dst=/app/examples/ex_7DBAS-S ghcr.io/kiguli/impact:main examples/ex_7DBAS-S/run_benchmark.sh