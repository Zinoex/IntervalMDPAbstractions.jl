#! /bin/bash

apptainer exec --bind $PWD/systems/IMPaCT/ex_syscore_running_example-RA:/app/examples/ex_syscore_running_example-RA ../impact_container.sif examples/ex_syscore_running_example-RA/apptainer_run.sh