#! /bin/bash

docker run --mount type=bind,src=$PWD/systems/IMPaCT/ex_2Drobot-RA-U,dst=/app/examples/ex_2Drobot-RA-U ghcr.io/kiguli/impact:main examples/ex_2Drobot-RA-U/run_benchmark.sh
