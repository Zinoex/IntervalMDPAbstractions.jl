#! /bin/bash

apptainer exec --cwd=/app --bind $PWD/systems/IMPaCT/ex_4DBAS-S:/app/examples/ex_4DBAS-S $PWD/systems/IMPaCT/impact_main.sif /app/examples/ex_4DBAS-S/run_benchmark.sh