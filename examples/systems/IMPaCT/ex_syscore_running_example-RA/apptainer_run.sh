#! /bin/bash

apptainer exec --cwd=/app --bind $PWD/systems/IMPaCT/ex_syscore_running_example-RA:/app/examples/ex_syscore_running_example-RA $PWD/systems/IMPaCT/impact_main.sif /app/examples/ex_syscore_running_example-RA/run_benchmark.sh