#! /bin/bash

docker run --mount type=bind,src=$PWD/systems/IMPaCT/ex_car_parking-RA,dst=/app/examples/ex_car_parking-RA ghcr.io/kiguli/impact:main examples/ex_car_parking-RA/run_benchmark.sh