#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:51:38 2019

@author: jwitte
"""

import argparse
from argparse import RawTextHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
import operator
import re

from find_files_per_regex import find_files
from orgmode_to_csv import org_to_csv
from orgmode_to_csv import csv_to_nparray
from itertools import compress

plt.rcParams["figure.figsize"] = (4.0, 4.2) # in inches

def parse_args():
    def path_type(name):
        path = os.path.abspath(name)
        assert os.path.exists(path), "{} does not exist.".format(path)
        return path

    parser = argparse.ArgumentParser(
            description='''Plot throughput [DoFs / s] against DoFs per node.''',
            formatter_class=RawTextHelpFormatter
            )
    parser.add_argument(
            '-dir', '--root_dir',
            type=path_type,
            default='.',
            help='The directory containing the folders with time measurements'
            )
    parser.add_argument(
            '-deg', '--degree',
            type=int,
            default=3,
            choices=[3, 7, 15],
            help='The polynomial degree'
            )
    args = parser.parse_args()
    return args


LINE_STYLES = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dashed', 'dashdotdotted']
LINE_WIDTH = 1.5
MARKER_SIZE = 8
MARKER_STYLES = ['.', '^', 'x', 'd', 's', 'v']
RANKS_PER_NODE = int(40) # bwunicluster2


def plot_throughput(root_dir,
                        str_section,
                        str_color='black',
                        do_inline_labels=False,
                        skip_plotting=False):
    eligible_testnames = ['ACP','MCP','AVP','MVP']

    eligible_names = ['AVS','MVS','ACS_DG','MCS_DG','AVS_DG','MVS_DG']

    test_to_linestyle = {
        name: style for name, style in zip(eligible_names, LINE_STYLES)
    }
    test_to_marker = {
        name: style for name, style in zip(eligible_names, MARKER_STYLES)
    }

    pattern = r'(Q|DGQ)(\d+)'
    match = re.search(pattern, str(root_dir))
    str_fem = match.group(1)
    n_degree = int(match.group(2))
    
    str_xlabel = r'\d+[kMG]DoFs'

    str_ylabel = r'apply (max)'

    str_tests = r'[AM][CV]P'

    str_subtests = r'(2560)prcs'

    pattern_file = r'{}_{}_{}_{}_3D_\d+deg_{}\.0019\.time\Z'.format(
        str_section,
        str_subtests,
        str_fem,
        str_tests,
        str_xlabel)

    orgfiles = [
        os.path.join(root_dir, basename)
        for basename in find_files(root_dir, pattern_file)
    ]
    # print(pattern_file,str_ylabel,orgfiles,sep='\n')
    
    def yield_testnames():
        pattern = str_tests
        for file in orgfiles:
            match = re.search(pattern, str(file))
            if match:
                yield match.group(0)

    def convert_metric_prefix(name):
        prefix_to_number = {'k': 1e+3, 'M': 1e+6, 'G': 1e+9}
        match = re.search(r'(\d+)([kMG])', name)
        assert match, "No valid prefix: {}".format(name)
        value = float(match.group(1)) * prefix_to_number[match.group(2)]
        return value
    
    def convert_method_prefix(name):
        return name

    testnames = set(yield_testnames())
    print('Testnames found: {}'.format(testnames))
    testnames = testnames.intersection(eligible_testnames)
    testnames = sorted(testnames, key=convert_method_prefix, reverse=False)
    print('Testnames used: {}'.format(testnames))

    fieldnames = [str_ylabel]

    def yield_orgfiles():
        for testname in testnames:
            def yield_per_test():
                for file in orgfiles:
                    fname = str(file)
                    if (fname.find('_{}'.format(testname)) != -1):
                        yield file
            filtered_files = set(yield_per_test())
            yield filtered_files
    orgfiles_per_test = list(yield_orgfiles())

    #: Extract x- and y-Data to be plotted
    def yield_xydata():
        for orgfiles in orgfiles_per_test:
            def yield_xy():
                for file in orgfiles:
                    match = re.search(str_xlabel, str(file))
                    submatch = re.search(str_subtests, str(file))
                    
                    n_dofs = convert_metric_prefix(match.group(0))
                    n_ranks = int(submatch.group(1))
                    n_nodes = n_ranks // RANKS_PER_NODE
                    
                    fname_csv = org_to_csv(file, fieldnames)
                    data = csv_to_nparray(fname_csv,
                                          dtype=np.double,
                                          delimiter=';',
                                          skip_header=1
                                          )
                    os.remove(fname_csv)
                    median = np.median(data, axis=0)
                    t_wall = median.tolist() # median of wall-time
                    
                    n_million_dofs = n_dofs / 1e6
                    x = n_dofs / n_nodes
                    y = n_million_dofs / t_wall

                    print("extracted (x, y) = ({}, {})".format(x, y))
                    yield x, y

            xy = list(yield_xy())
            xy.sort(key=operator.itemgetter(0)) # sort by x-value
            yield xy

    xydata = list(yield_xydata())

    #: Create (sub)plot
    plt.semilogx() # plt.loglog()
    plt.grid(True)

    #: Insert data
    for xy, testname in zip(xydata, testnames):
        x, y = zip(*xy)

        def convert_smoother_notation(testname):
            old_to_new = {'AVP': 'AVS', 'ACP': 'ACS', 'MCP': 'MCS', 'MVP': 'MVS'}
            return old_to_new[testname]

        name = convert_smoother_notation(testname)
        if str_fem == 'DGQ':
            name = name + r'_DG'
        
        plt.plot(x,
                 y,
                 label=name,
                 color=str_color,
                 marker=test_to_marker[name],
                 markersize=MARKER_SIZE,
                 linewidth=LINE_WIDTH)

        if do_inline_labels:
            if len(x) is 1: # inline labels
                plt.text(0.5*x[0], y[0], str(name), fontsize='small')
            else:
                plt.text(1.25*x[0], y[0], str(name), fontsize='small')
                
    #: Legend
    plt.legend()
    return xydata  # todo


def set_xticks(xydata):
    xticks = set()
    for xy in xydata:
        x, y = zip(*xy)
        xticks = xticks.union(set(x))
    xticks = sorted(xticks)
    xticklabels = [str(value) for value in xticks]
    plt.xticks(ticks=xticks, labels=xticklabels)
    plt.tick_params(axis='x', which='minor', bottom=False)


def main():
    options = parse_args()
    degree = options.degree
    fig = plt.figure()

    root_dir = options.root_dir

    # def make_folder_name(str_fem, str_degree, str_section=None):
    #     name = r'{}{}'.format(str_fem, str_degree)
    #     if str_section:
    #         name = name + r'_' + str_section
    #     return name

    ## Plot results for k = 3
    Qk_solve_dir = os.path.join(root_dir, f'Q{degree}_solve')
    DGQk_solve_dir = os.path.join(root_dir, f'DGQ{degree}_solve')
    
    ax11 = plt.subplot(111)
    # plt.title("AVS")
    plt.xlabel("DoFs per node")
    plt.ylabel("Throughput [million DoFs / s]")
    section = r'solve'

    xydata = plot_throughput(Qk_solve_dir,
                             str_section=section)
    xydata = plot_throughput(DGQk_solve_dir,
                             str_color='limegreen',
                             str_section=section)

    # set_xticks(xydata)
    # plt.suptitle("Strong Scaling ({})".format(method))
    handles, labels = ax11.get_legend_handles_labels()
    n_labels = len(labels)
    # fig.legend(handles, labels, loc='lower left', ncol=n_labels, mode="expand")
    fig.tight_layout()
    plt.show()

    return 0


if __name__ == '__main__':
    main()
