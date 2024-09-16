#! /bin/bash

docker run -it --mount type=bind,src=$PWD/../ex_syscore_running_example-RA,dst=/app/examples/ex_syscore_running_example-RA ghcr.io/kiguli/impact:main 

# "cd examples/ex_2Drobot-R-U/; make; ./robot2D"