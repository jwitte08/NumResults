import argparse
from argparse import RawTextHelpFormatter
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='''parsing data and counting FLOPs afterwards'''
                                     , formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('-v','--verbose'
                        , action='store_true'
                        , help='activates verbose output'
    )
    parser.add_argument('parentdir'
                        , help='source directory of files'
    )
    args = parser.parse_args()
    return args

def main():
    options = parse_args()
    parentdir = os.path.abspath(options.parentdir)
    assert (os.path.isdir(parentdir)),'Source directory is invalid.'
    
    # parse likwid-perfctr's output
    parse_likwid = os.path.expanduser(r'~/scripts/likwid_parse/parse_csv.py')
    if options.verbose:
        subprocess.run(['python',parse_likwid,parentdir,r'-E','FP_ARITH','--verbose'])
    else:
        subprocess.run(['python',parse_likwid,parentdir,r'-E','FP_ARITH'])

    # parse runlog for misc. information
    parse_runlog = os.path.expanduser(r'~/scripts/likwid_parse/parse_rlog.py')
    if options.verbose:
        subprocess.run(['python',parse_runlog,parentdir,'--verbose'])
    else:
        subprocess.run(['python',parse_runlog,parentdir])

    # compute total FLOPs
    dirs = (
        os.path.join(parentdir,dir)
        for dir in os.listdir(parentdir)
        if os.path.isdir(os.path.join(parentdir,dir))
    )
    count_flops = os.path.expanduser(r'~/scripts/likwid_parse/flops2csv.py')
    for d in dirs:
        if options.verbose:
            subprocess.run(['python',count_flops,d,'--verbose'])
        else:
            subprocess.run(['python',count_flops,d])
        
if __name__ == '__main__':
    main()
