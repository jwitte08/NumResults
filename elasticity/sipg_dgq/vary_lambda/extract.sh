#!/bin/bash

DEGREE=7

for SMO in "ACP" "MCP" "AVP" "MVP"
do
    echo ""
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "Smoother: $SMO"
    for LAMBDA in "1.000e+00" "1.000e+01" "5.000e+01" "1.000e+02" "1.000e+03" "1.000e+04"
    do
	echo ""
	echo "Lambda = $LAMBDA"
	for VARIANT in "exact" "fast" "diag"
	do
	    echo "Local solver: $VARIANT"
	    grep "of iterations" ${SMO}_2D_${DEGREE}deg_${VARIANT}_2steps_1.000e+00mu_${LAMBDA}lambda1.000e+01ip.log
	done
    done
done

#ACP_2D_3deg_fast_2steps_1.000e+00mu_5.000e+01lambda1.000e+01ip.log

