'''
Created on Apr 08, 2019

work on metrics

@author: jwitte
'''

import argparse
from argparse import RawTextHelpFormatter
import os
import subprocess
import numpy as np

from csvutil import csvUtility
from pathcollect import rcseek_paths
from pathcollect import sort_by_metaprm

PWD_ = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(
        description="""work on metrics""",
        formatter_class=RawTextHelpFormatter
    )
    def abspath(fname):
        return os.path.abspath(fname)
    parser.add_argument('filename',
                        type=abspath,
                        help="regular expression identifying the input files"
    )
    def cspair(string):
        first,second = string.split(',')
        return int(first),int(second)
    parser.add_argument('-div','--table-divider',
                        type=cspair,
                        default='1,1',
                        metavar='pair',
                        help="""comma-separated integer pair dividing the CSV files into its typical
blocks description, observational description, feature description and
plain data, where the first number is the row and the second is the"""
    )
#     def csint(string):
#         return [int(s) for s in string.split(',')]
#     parser.add_argument('-cols','--selected-columns',
#                         type=csint,
#                         default=[],
#                         metavar='tuple',
#                         help="""comma-separated tuple of integers used to extract data from selected
# columns"""
#     )
    args = parser.parse_args()
    return args

def main():
    options = parse_args()
    fpath_in = options.filename
    dirpath,fname_in = os.path.split(fpath_in)
    fname_out = 'table_{}.tex'.format(fname_in.rstrip(r'.csv'))
    fpath_out = os.path.join(dirpath,fname_out)
    
    util = csvUtility()
    divrow,divcol = options.table_divider
    table = util.disassemble(fpath_in,data_type=int,n_rows=divrow,n_cols=divcol)
    desc,obs,feat,data = table

    #: convert
    def csnumber (number):
        """comma-separated number between thousands"""
        return ("{:,}".format(number))
    tmp = data/1e+6
    data = tmp.astype(int)
    data_str = np.empty(data.shape,dtype='U32')
    n_rows,n_cols = data.shape
    for m in range(n_rows):
        for n in range(n_cols):
            data_str[m,n] = csnumber(data[m,n])
    print (data_str)

    fpath_tmp = os.path.join(PWD_,'tmp.csv')
    with open(fpath_tmp,'w') as fw:
        util.assemble(fw,desc,obs,feat,data_str)
    csv2table = os.path.expanduser(r'~/scripts/likwid_parse/csv2table.py')
    subprocess.run(['python3',csv2table,fpath_tmp,'-O',fpath_out,'--head-rows',str(divrow)])
    os.remove(fpath_tmp)
    
if __name__ == '__main__':
    main()
