#! /bin/bash

apptainer exec --cwd=/app --bind $PWD/systems/IMPaCT/ex_car_parking-RA:/app/examples/ex_car_parking-RA $PWD/systems/IMPaCT/impact_main.sif /app/examples/ex_car_parking-RA/run_benchmark.sh