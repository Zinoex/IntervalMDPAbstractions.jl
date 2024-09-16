#! /bin/bash

apptainer exec --cwd=/app --bind $PWD/systems/IMPaCT/ex_2Drobot-RA-U:/app/examples/ex_2Drobot-RA-U $PWD/systems/IMPaCT/impact_main.sif /app/examples/ex_2Drobot-RA-U/run_benchmark.sh