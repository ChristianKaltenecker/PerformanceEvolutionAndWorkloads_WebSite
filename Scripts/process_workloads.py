#!/bin/env python3

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
from typing import Dict, List, Tuple

WORKLOADS: Dict[str, List[str]] = {
    'tar': ['enwik9', 'linux_kernel', 'hmdb', '3d_modelle',
            'map_of_countries_borders', 'eu_es_male', 'davis'],
    'tar_compress': ['enwik9', 'linux_kernel', 'hmdb', '3d_modelle',
                     'map_of_countries_borders', 'eu_es_male', 'davis'],
    'tar_extract': ['enwik9', 'linux_kernel', 'hmdb', '3d_modelle',
                    'map_of_countries_borders', 'eu_es_male', 'davis'],
    'z3': ['LRA', 'QF_FP', 'QF_LRA', 'QF_UFLRA'],
    'FastDownward':
        ['airport_p07airport2p2', 'barmanopt11strips_pfile01001', 'blocks_probBLOCKS100', 'datanetworkopt18strips_p13',
         'depot_p10', 'driverlog_p08', 'elevatorsopt08strips_p24', 'elevatorsopt11strips_p20',
         'floortileopt11strips_optp01002', 'freecell_prob45', 'gedopt14strips_d76', 'grid_prob02', 'gripper_prob07',
         'hikingopt14strips_ptesting234', 'logistics00_probLOGISTICS71', 'logistics98_prob31', 'miconic_s132',
         'movie_prob29', 'mprime_prob02', 'mystery_prob30', 'nomysteryopt11strips_p17', 'openstacksopt08strips_p12',
         'openstacksopt11strips_p07', 'openstacksopt14strips_p20_3', 'openstacksstrips_p07',
         'organicsynthesisopt18strips_p09', 'organicsynthesissplitopt18strips_p01', 'parcprinter08strips_p25',
         'parcprinteropt11strips_p10', 'pathways_p04', 'pegsol08strips_p25', 'pegsolopt11strips_p15',
         'pipesworldnotankage_p12net2b10g4', 'pipesworldtankage_p31net4b14g3t20', 'psrsmall_p48s101n5l3f30',
         'rovers_p07', 'satellite_p06pfile6', 'scanalyzer08strips_p06', 'scanalyzeropt11strips_p12',
         'snakeopt18strips_p04', 'sokobanopt08strips_p22', 'sokobanopt11strips_p17', 'storage_p14',
         'termesopt18strips_p18', 'tetrisopt14strips_p024', 'tidybotopt11strips_p01', 'tpp_p06',
         'transportopt08strips_p04', 'transportopt11strips_p06', 'transportopt14strips_p14', 'trucksstrips_p08',
         'visitallopt11strips_problem06full', 'visitallopt14strips_p057', 'woodworkingopt08strips_p24',
         'woodworkingopt11strips_p05', 'zenotravel_p11']
}

NON_FUNCTIONAL_PROPERTIES: Dict[str, List[Tuple[str, str]]] = {
    'tar': [('compress_performance', 'Compression'), ('extract_performance', 'Extraction')],
    'z3': [('performance', 'Execution Time'), ('memory', 'Memory Consumption')],
    'FastDownward': [('performance', 'Reported Performance'), ('intmemory', 'Memory Consumption'),
                     ('logSize', 'Size of Log File')]
}

WORKLOAD_COLUMN_NAME = 'workload'


def print_usage():
    print("./main.py <MeasurementDirectory> <OutputDirectory>")
    print("MeasurementDirectory\t the directory containing the "
          "measurement data in different folders")
    print("OutputDirectory\t\t the directory to save the plots to")


def visualize_workloads(input_directory: str, output_directory: str) -> None:
    for case_study in WORKLOADS.keys():
        print(case_study)
        case_study_directory = os.path.join(input_directory, case_study)
        # Some error handling
        if not os.path.exists(case_study_directory) or not os.path.isdir(case_study_directory):
            print(f'The case study directory does not exist: {case_study_directory}')
            continue
        # Parse the data
        dataframe: pd.DataFrame = pd.read_csv(os.path.join(case_study_directory, 'measurements.csv'), sep=';')
        case_study_output_directory = os.path.join(output_directory, case_study)

        # Create a new directory for the case study if not already included
        if not os.path.exists(case_study_output_directory):
            try:
                os.mkdir(case_study_output_directory)
            except OSError:
                print(f'Creation of the directory {case_study_output_directory} failed')
                exit(-1)
        create_scatter_plot(dataframe, case_study, case_study_output_directory)


def convert_measurements_file(data: pd.DataFrame, case_study: str) -> pd.DataFrame:
    data[WORKLOAD_COLUMN_NAME] = [0] * len(data.index)
    # Add an additional column for the different workloads
    for workload_counter in range(0, len(WORKLOADS[case_study])):
        workload = WORKLOADS[case_study][workload_counter]
        data.loc[data[workload] == "1", WORKLOAD_COLUMN_NAME] = workload
        data = data.drop(columns=[WORKLOADS[case_study][workload_counter]], axis=1)
    data = data.drop(columns=["workloads"], axis=1)
    return data


def create_scatter_plot(data: pd.DataFrame, case_study: str, output_directory: str) -> None:
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.figure()
    data = convert_measurements_file(data, case_study)

    sns.scatterplot(x=NON_FUNCTIONAL_PROPERTIES[case_study][0][0], y=NON_FUNCTIONAL_PROPERTIES[case_study][1][0],
                    data=data, style=WORKLOAD_COLUMN_NAME, hue=WORKLOAD_COLUMN_NAME)
    plt.xlabel(NON_FUNCTIONAL_PROPERTIES[case_study][0][1])
    plt.ylabel(NON_FUNCTIONAL_PROPERTIES[case_study][1][1])
    plt.legend(title='Workload', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(output_directory, 'scatterplot_all.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure()
    sns.violinplot(x=WORKLOAD_COLUMN_NAME, y=NON_FUNCTIONAL_PROPERTIES[case_study][0][0], data=data)
    plt.xlabel(WORKLOAD_COLUMN_NAME)
    plt.ylabel(NON_FUNCTIONAL_PROPERTIES[case_study][0][1])
    plt.savefig(os.path.join(output_directory, 'scatterplot_workloads.pdf'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print_usage()
        exit(-1)
    input_directory: str = sys.argv[1]
    output_directory: str = sys.argv[2]
    if not os.path.exists(input_directory) or not os.path.isdir(input_directory) \
            or not os.path.exists(output_directory) or not os.path.isdir(output_directory):
        print("One of the paths is not a valid path or directory! Aborting...")
        exit(-1)
    visualize_workloads(input_directory, output_directory)
