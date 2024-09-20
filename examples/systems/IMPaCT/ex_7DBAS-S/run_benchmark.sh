#! /bin/bash

cd examples/ex_7DBAS-S/
make
./BAS7D

rm is.h5 ts.h5 ss.h5
rm maxatm.h5 minatm.h5
rm maxtm.h5 mintm.h5
rm BAS7D