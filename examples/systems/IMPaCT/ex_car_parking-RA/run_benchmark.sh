#! /bin/bash

cd examples/ex_car_parking-RA/
make
./car_parking

rm is.h5 ts.h5 ss.h5
rm maxatm.h5 minatm.h5
rm maxttm.h5 minttm.h5
rm maxtm.h5 mintm.h5
rm car_parking