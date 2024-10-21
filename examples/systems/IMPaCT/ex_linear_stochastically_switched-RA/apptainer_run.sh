#! /bin/bash

apptainer exec --cwd=/app --bind $PWD/systems/IMPaCT/ex_linear_stochastically_switched-RA:/app/examples/ex_linear_stochastically_switched-RA $PWD/systems/IMPaCT/impact_main.sif /app/examples/ex_linear_stochastically_switched-RA/run_benchmark.sh