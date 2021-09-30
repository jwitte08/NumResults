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

plt.rcParams["figure.figsize"] = (8,8.4)

def parse_args():
    def path_type(name):
        path = os.path.abspath(name)
        assert os.path.exists(path), "{} does not exist.".format(path)
        return path

    parser = argparse.ArgumentParser(
            description='''Plot strong scaling results''',
            formatter_class=RawTextHelpFormatter
            )
    parser.add_argument(
            '-dir', '--root_dir',
            type=path_type,
            default='.',
            help='The directory containing the strong scaling results'
            )
    parser.add_argument(
            '-adir', '--root_adir',
            type=path_type,
            default='.',
            help='Another directory containing the strong scaling results'
            )
    parser.add_argument(
            '-mth', '--method',
            type=str,
            default='MVP',
            choices=['ACP', 'MCP', 'AVP', 'MVP'],
            help='The smoothing variant'
            )
    parser.add_argument(
            '-fem', '--element',
            type=str,
            default='DGQ',
            choices=['DGQ', 'Q'],
            help='The finite element method'
            )
    args = parser.parse_args()
    return args


def plot_strong_scaling(str_method, str_section, str_fem, str_color='black', do_perf=True, do_inline_labels=True, this_ylabel=None, this_dir=None, other_xydata=None, skip_plotting=False):
    eligible_names = ['2MDoFs', '16MDoFs', '134MDoFs', '1GDoFs', '8GDoFs'] # DGQ3
    eligible_names = ['16MDoFs', '134MDoFs', '1GDoFs', '8GDoFs'] # DGQ7_solve
    # eligible_names = ['912kDoFs', '7MDoFs', '57MDoFs', '454MDoFs', '3GDoFs'] #, '29GDoFs'] # Q3
    # eligible_names = ['11MDoFs', '90MDoFs', '721MDoFs', '5GDoFs'] # Q7
    # eligible_names = ['13MDoFs', '111MDoFs', '887MDoFs', '7GDoFs', '56GDoFs'] # Q15
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dashed', 'dashdotdotted']
    test_to_linestyle = {
            name: style for name, style in zip(eligible_names, linestyles)
            }
    markerstyles = ['.', '^', 'x', 'd', 's', 'v']
    test_to_marker = {
            name: style for name, style in zip(eligible_names, markerstyles)
            }
    marker_size = 8
    line_width = 1.5

    options = parse_args()
    root_dir = options.root_dir
    if this_dir:
        root_dir = this_dir
    str_xlabel = r'(\d+)prcs'
    str_ylabel = r'apply (max)'
    if this_ylabel:
        str_ylabel = this_ylabel
    str_tests = r'\d+[kMG]DoFs'
    pattern_file = r'{}_{}_{}_{}_3D_\d+deg_{}\.0019\.time\Z'.format(
            str_section,
        str_xlabel,
        str_fem,
            str_method,
            str_tests
            )

    orgfiles = [
            os.path.join(root_dir, basename)
            for basename in find_files(root_dir, pattern_file)
            ]
    print(pattern_file,str_ylabel,orgfiles,sep='\n')
    
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

    testnames = set(yield_testnames())
    print('Testnames found: {}'.format(testnames))
    testnames = testnames.intersection(eligible_names)
    testnames = sorted(testnames, key=convert_metric_prefix, reverse=False)
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
                    n_procs = int(match.group(1))

                    fname_csv = org_to_csv(file, fieldnames)
                    data = csv_to_nparray(fname_csv,
                                          dtype=np.double,
                                          delimiter=';',
                                          skip_header=1
                                          )
                    os.remove(fname_csv)
                    median = np.median(data, axis=0)
                    print("extracted (x, y) = ({}, {})".format(n_procs, median.tolist()))
                    yield n_procs, median.tolist()

            xy = list(yield_xy())
            xy.sort(key=operator.itemgetter(0))
            yield xy

    xydata = list(yield_xydata())

    if skip_plotting:
        return xydata
    
    #: Create (sub)plot
    plt.loglog()
    plt.grid(True)

    #: Insert data
    if not other_xydata:
        for xy, name in zip(xydata, testnames):
            x, y = zip(*xy)

            def yield_perfect():
                first, *rest = y
                n = len(x)
                for i in range(n):
                    yield first * (0.5**i)
            y_perf = list(yield_perfect())
            plt.plot(x, y,
                     label=name,
                     color=str_color,
                     marker=test_to_marker[name],
                     markersize=marker_size,
                     linewidth=line_width
                     )
            # linestyle=test_to_linestyle[name],
            if do_inline_labels:
                if len(x) is 1: # inline labels
                    plt.text(0.5*x[0], y[0], str(name), fontsize='small')
                else:
                    plt.text(1.25*x[0], y[0], str(name), fontsize='small')

            if do_perf:
                plt.plot(x, y_perf, color='grey', linestyle='dotted', linewidth='0.8')
    else:
        for xy, other_xy, name in zip(xydata, other_xydata, testnames):
            x, y = zip(*xy)
            other_x, other_y = zip(*other_xy)
            xmatches = [i in other_x for i in x]
            x = list(compress(x, xmatches))
            y = list(compress(y, xmatches))
            plt.plot(x, y,
                     label=name,
                     color=str_color,
                     marker=test_to_marker[name],
                     markersize=marker_size,
                     linewidth=line_width
                     )
                
    #: Legend
    # plt.legend()
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
    method = options.method
    fem = options.element
    fig = plt.figure()

    ax11 = plt.subplot(221)
    plt.title("ACS")
    plt.ylabel("Wall time [s]")
    section = r'solve'
    xydata = plot_strong_scaling(str_method='ACP', str_section=section, str_fem=fem, skip_plotting=True)
    section = r'smooth'
    xydata_smooth = plot_strong_scaling(str_method='ACP',
                                        str_section=section,
                                        str_fem=fem,
                                        str_color='deepskyblue',
                                        do_perf=False,
                                        do_inline_labels=False,
                                        this_ylabel=r'setup (max)',
                                        this_dir=options.root_adir,
                                        other_xydata=xydata)
    section = r'solve'
    xydata = plot_strong_scaling(str_method='ACP', str_section=section, str_fem=fem)

    plt.subplot(222, sharex=ax11)
    plt.title("MCS")
    plt.xlabel("Number of cores")
    section = r'solve'
    xydata = plot_strong_scaling(str_method='MCP', str_section=section, str_fem=fem, skip_plotting=True)
    section = r'smooth'
    xydata_smooth = plot_strong_scaling(str_method='MCP',
                                        str_section=section,
                                        str_fem=fem,
                                        str_color='deepskyblue',
                                        do_perf=False,
                                        do_inline_labels=False,
                                        this_ylabel=r'setup (max)',
                                        this_dir=options.root_adir,
                                        other_xydata=xydata)
    section = r'solve'
    xydata = plot_strong_scaling(str_method='MCP', str_section=section, str_fem=fem)

    plt.subplot(223, sharex=ax11)
    plt.title("MVS")
    section = r'solve'
    xydata = plot_strong_scaling(str_method='MVP', str_section=section, str_fem=fem, skip_plotting=True)
    section = r'smooth'
    xydata_smooth = plot_strong_scaling(str_method='MVP',
                                        str_section=section,
                                        str_fem=fem,
                                        str_color='deepskyblue',
                                        do_perf=False,
                                        do_inline_labels=False,
                                        this_ylabel=r'setup (max)',
                                        this_dir=options.root_adir,
                                        other_xydata=xydata)
    section = r'solve'
    xydata = plot_strong_scaling(str_method='MVP', str_section=section, str_fem=fem)
    plt.ylabel("Wall time [s]")
    plt.xlabel("Number of cores")

    set_xticks(xydata)
    # plt.suptitle("Strong Scaling ({})".format(method))
    handles, labels = ax11.get_legend_handles_labels()
    n_labels = len(labels)
    # fig.legend(handles, labels, loc='lower left', ncol=n_labels, mode="expand")
    fig.tight_layout()
    plt.show()

    return 0


if __name__ == '__main__':
    main()
