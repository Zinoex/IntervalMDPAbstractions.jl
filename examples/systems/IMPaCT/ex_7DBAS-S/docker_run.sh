#! /bin/bash

docker run -it --mount type=bind,src=$PWD/../ex_7DBAS-S,dst=/app/examples/ex_7DBAS-S ghcr.io/kiguli/impact:main 

# "cd examples/ex_2Drobot-R-U/; make; ./robot2D"