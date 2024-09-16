#! /bin/bash

cd examples/ex_syscore_running_example-RA/
make
./running_example

rm is.h5 ts.h5 ss.h5
rm maxatm.h5 minatm.h5
rm maxttm.h5 minttm.h5
rm maxtm.h5 mintm.h5
rm running_example