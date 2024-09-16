#! /bin/bash

apptainer exec --bind $PWD/systems/IMPaCT/ex_2Drobot-RA-U:/app/examples/ex_2Drobot-RA-U ../impact_container.sif examples/ex_2Drobot-RA-U/apptainer_run.sh