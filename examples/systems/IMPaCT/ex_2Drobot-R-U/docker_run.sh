#! /bin/bash

docker run --mount type=bind,src=$PWD/systems/IMPaCT/ex_2Drobot-R-U,dst=/app/examples/ex_2Drobot-R-U ghcr.io/kiguli/impact:main examples/ex_2Drobot-R-U/run_benchmark.sh