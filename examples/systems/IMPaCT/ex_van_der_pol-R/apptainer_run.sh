#! /bin/bash

apptainer exec --cwd=/app --bind $PWD/systems/IMPaCT/ex_van_der_pol-R:/app/examples/ex_van_der_pol-R $PWD/systems/IMPaCT/impact_main.sif /app/examples/ex_van_der_pol-R/run_benchmark.sh