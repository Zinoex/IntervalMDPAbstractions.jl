#! /bin/bash

apptainer exec --bind $PWD/systems/IMPaCT/ex_4DBAS-S:/app/examples/ex_4DBAS-S ../impact_container.sif examples/ex_4DBAS-S/apptainer_run.sh