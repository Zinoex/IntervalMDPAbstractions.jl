#! /bin/bash

docker run -it --mount type=bind,src=$PWD/../ex_2Drobot-R-U,dst=/app/examples/ex_2Drobot-R-U ghcr.io/kiguli/impact:main 

# "cd examples/ex_2Drobot-R-U/; make; ./robot2D"