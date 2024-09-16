#! /bin/bash

apptainer exec --bind $PWD/systems/IMPaCT/ex_van_der_pol-R:/app/examples/ex_van_der_pol-R ../impact_container.sif examples/ex_van_der_pol-R/apptainer_run.sh