#! /bin/bash

apptainer exec --cwd=/app --bind $PWD/systems/IMPaCT/ex_7DBAS-S:/app/examples/ex_7DBAS-S $PWD/systems/IMPaCT/impact_main.sif /app/examples/ex_7DBAS-S/run_benchmark.sh