'''
Created on Apr 15, 2019

@author: jwitte
'''

import sys
import os
import re
import numpy as np
from pathcollect import rcseek_paths
#from pathcollect import sort_by_metaprm
from csvutil import csvUtility
from csvutil import extract_columns

PWD_ = os.getcwd()

def complexity_constants():
    """computes the constants of the operations' complexity"""
    smo_str = 'AVP'
    fpath_flop,*_ = rcseek_paths(PWD_,'METRIC_{}'.format(smo_str),fflag=True)
    fpath_rlog,*_ = rcseek_paths(PWD_,'RUNLOG_{}'.format(smo_str),fflag=True)
    util = csvUtility()
    tab_flop = util.disassemble(fpath_flop,data_type=int,n_rows=3)
    tab_rlog = util.disassemble(fpath_rlog,data_type=int,n_rows=3)
    _,sections,feat,flops = tab_flop
    _,_,_,counts = tab_rlog
    print (feat)
    def gen_degree():
        for dstr in feat[1]:
            match = re.search('DEG(\d+)',dstr)
            yield int(match.group(1))
    degrees = list(gen_degree())
    print (degrees)
    def gen_counttuple():
        rest = list(counts[0])
        while (rest):
            n_cells,n_subds,*rest = rest
            yield n_cells,n_subds
    ctuples = list(gen_counttuple())
    constants = np.empty(flops.shape,dtype=int)
    n_rows,n_cols = flops.shape
    for m in range(n_rows):
        sect,*_ = sections[m]
        for n in range(n_cols):
            n_flops = flops[m][n]
            n_cells,n_subds = ctuples[n]
            degree = degrees[n]
            def get_constant():
                dim = 3
                if ('MF' in sect or 'residual' in sect):
                    res = n_flops/n_cells
                elif ('TPSS' in sect):
                    res = n_flops/n_subds
                else:
                    assert false,"..."
                if ('vmult' in sect or 'step' in sect or 'residual' in sect or 'correction' in sect):
                    res = res/(degree+1)**(dim+1)
                elif ('compute' in sect):
                    res = res/(degree+1)**3
                else:
                    assert false,"..."
                return res
            constants[m][n] = get_constant()
    head = np.array([['degree']])
    obs = sections
    feat = np.array([degrees])
    print (head)
    print (sections)
    print (feat)
    fpath_out = os.path.abspath('cmplxconstant_{}.csv'.format(smo_str))
    print (constants)
    with open(fpath_out,'w') as fw:
        util.assemble(fw,head,sections,feat,constants)
    
def main():
    complexity_constants()
    
if __name__ == '__main__':
    main()
