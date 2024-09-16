#! /bin/bash

docker run -it --mount type=bind,src=$PWD/../ex_van_der_pol-R,dst=/app/examples/ex_van_der_pol-R ghcr.io/kiguli/impact:main 

# "cd examples/ex_2Drobot-R-U/; make; ./robot2D"