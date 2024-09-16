#! /bin/bash

docker run --mount type=bind,src=$PWD/../ex_syscore_running_example-RA,dst=/app/examples/ex_syscore_running_example-RA ghcr.io/kiguli/impact:main examples/ex_syscore_running_example-RA/run_benchmark.sh