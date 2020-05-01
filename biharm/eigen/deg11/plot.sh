#!/bin/bash

python3 ../../scripts/plot_biharmeigenfunctions.py -N 21 -grd "6x4" \
	-spc 'Bip_wrt_sqL,Eigenfunctions of $B_{ip} v_i = \lambda_i L^2 v_i$' \
	'L_wrt_M,Eigenfunctions of $L v_i = \lambda_i M v_i$' \
	'Bip_wrt_M,Eigenfunctions of $B_{ip} v_i = \lambda_i M v_i$' \
	'Bip,Eigenfunctions of $B_{ip} v_i = \lambda_i v_i$' \
	'B,Eigenfunctions of $B v_i = \lambda_i v_i$' \
	'L,Eigenfunctions of $L v_i = \lambda_i v_i$' \
	'sqL,Eigenfunctions of $L^2 v_i = \lambda_i v_i$' \
	'M,Eigenfunctions of $M v_i = \lambda_i v_i$' \
