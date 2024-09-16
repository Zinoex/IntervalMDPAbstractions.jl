#! /bin/bash

apptainer exec --cwd=/app --bind $PWD/systems/IMPaCT/ex_2Drobot-R-U:/app/examples/ex_2Drobot-R-U $PWD/systems/IMPaCT/impact_main.sif /app/examples/ex_2Drobot-R-U/run_benchmark.sh