#! /bin/bash

apptainer exec --bind $PWD/systems/IMPaCT/ex_7DBAS-S:/app/examples/ex_7DBAS-S ../impact_container.sif examples/ex_7DBAS-S/apptainer_run.sh