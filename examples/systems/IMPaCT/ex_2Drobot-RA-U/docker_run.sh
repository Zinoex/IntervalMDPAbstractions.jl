#! /bin/bash

docker run -it --mount type=bind,src=$PWD/../ex_2Drobot-RA-U,dst=/app/examples/ex_2Drobot-RA-U ghcr.io/kiguli/impact:main 

# "cd examples/ex_2Drobot-R-U/; make; ./robot2D"