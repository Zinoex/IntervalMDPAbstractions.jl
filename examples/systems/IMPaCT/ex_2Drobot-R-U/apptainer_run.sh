#! /bin/bash

apptainer exec --bind $PWD/systems/IMPaCT/ex_2Drobot-R-U:/app/examples/ex_2Drobot-R-U ../impact_container.sif examples/ex_2Drobot-R-U/apptainer_run.sh