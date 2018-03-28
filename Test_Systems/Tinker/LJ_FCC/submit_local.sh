#!/bin/bash
#SUBMIT THE BENZENE JOB TO BE RUN


TINKER_PARAMS='\/home\/nabraham\/tinker\/params'

sed -i "s/PATHTOPARAMETERS/"$TINKER_PARAMS"/g" keyfile.key

Run_LatticeDynamics.py -i input.inp

sed -i "s/"$TINKER_PARAMS"/PATHTOPARAMETERS/g" keyfile.key

