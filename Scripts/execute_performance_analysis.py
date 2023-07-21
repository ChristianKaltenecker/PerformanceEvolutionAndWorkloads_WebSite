#!/bin/env python3
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from PerformanceEvolution.persisting_regression_analysis import PersistingRegressionAnalysis
from PerformanceEvolution.precision_analyzer import PrecisionAnalyzer
from PerformanceEvolution.workload_frequency_analyzer import WorkloadFrequencyAnalyzer
from PerformanceEvolution.workload_sensitivity import WorkloadSensitivityAnalyzer
from case_study import CaseStudy
from configuration_level import ConfigurationLevel
from option_level import OptionLevel
from recall_analyzer import RecallAnalyzer

NFP = "performance"  # (execution time in our case)
FM = "FeatureModel.xml"
Measurements = "measurements.csv"
Deviations = "deviations.csv"

AnalysisLevels = [
    ConfigurationLevel(),
    OptionLevel()
]


def print_usage() -> None:
    """
    Prints the usage of the python script.
    """
    print("Usage: execute_performance_analysis.py <InputPath> <OutputPath>")
    print("InputPath\t The path to the directory containing all relevant information of the case studies.")
    print("OutputPath\t The path to the directory where all plots should be exported to.")


def list_directories(path: str) -> List:
    """
    Returns the subdirectories of the given path.
    :param path: the path to find the subdirectories from.
    :return: the subdirectories as list.
    """
    for root, dirs, files in os.walk(path):
        return list(filter(lambda x: not x.startswith("."), dirs))


def create_directory(path: str) -> None:
    """
    Creates the given directory if it does not exist already.
    :param path: the path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def main() -> None:
    """
    The main method reads in the data of the case studies and evaluates the data with regard to the different
    research questions (1-4) of the study.
    """
    if len(sys.argv) != 3:
        print_usage()
        exit(0)

    # Read in the path to the case study data
    input_path = sys.argv[1]

    # Read in the output path of the plots
    output_path = sys.argv[2]

    fontsize = 30
    plt.rcParams.update({'font.size': fontsize})
    sns.set_style("whitegrid")

    case_studies = list_directories(input_path)
    print("Progress:")
    i = -1

    # In the next lines, we execute the performance change analysis at the configuration level and the option level
    for al in AnalysisLevels:
        if not os.path.exists(os.path.join(output_path, al.name)):
            os.mkdir(os.path.join(output_path, al.name))
        al.initialize_for_metrics(os.path.join(output_path, al.name))

    for case_study in case_studies:
        i += 1
        print(case_study + " (" + str(int((float(i) / len(case_studies)) * 100)) + "%)")
        # Read in one case study (i.e., its FM and measurements) after another (and wipe the data to save some RAM)
        cs = CaseStudy(case_study, os.path.join(input_path, case_study, FM),
                       os.path.join(input_path, case_study, Measurements),
                       os.path.join(input_path, case_study, Deviations))

        rq_count = 0
        for al in AnalysisLevels:
            if not os.path.exists(os.path.join(output_path, al.name, case_study)):
                os.mkdir(os.path.join(output_path, al.name, case_study))
            rq_count += 1
            print("\t" + al.get_name() + "...", end="")
            sys.stdout.flush()
            al.prepare(cs, input_path)
            al.evaluate_metrics(cs, os.path.join(output_path, al.name), input_path)
            al.generate_plots(cs, os.path.join(output_path, al.name, case_study), input_path)
            print("Finished!")

        for al in AnalysisLevels:
            al.finish(os.path.join(output_path, al.name), os.path.join(input_path, case_study))

    # Next, execute the analysis for precision, recal, workload sensitivity, persisting regressions and the workload
    # frequency
    for case_study in case_studies:
        cs = CaseStudy(case_study, os.path.join(input_path, case_study, FM),
                       os.path.join(input_path, case_study, Measurements),
                       os.path.join(input_path, case_study, Deviations))

        # Precision
        precision_analyzer = PrecisionAnalyzer()
        if not os.path.exists(os.path.join(output_path, "Precision")):
            os.mkdir(os.path.join(output_path, "Precision"))
        precision_analyzer.perform_analysis(os.path.join(output_path, "Precision"), cs,
                                            os.path.join(input_path, case_study, "models", "models.csv"),
                                            os.path.join(input_path, case_study))

        # Recall
        recall_analyzer = RecallAnalyzer()
        if not os.path.exists(os.path.join(output_path, "Recall")):
            os.mkdir(os.path.join(output_path, "Recall"))
        recall_analyzer.perform_analysis(os.path.join(output_path, "Recall"), cs,
                                         os.path.join(input_path, case_study, "models", "models.csv"),
                                         os.path.join(input_path, case_study))

        # Persisting regressions
        persisting_regression_analysis = PersistingRegressionAnalysis()
        if not os.path.exists(os.path.join(output_path, "PersistingRegressions")):
            os.mkdir(os.path.join(output_path, "PersistingRegressions"))
        persisting_regression_analysis.process_data(cs, os.path.join(input_path, case_study),
                                                    os.path.join(output_path, "PersistingRegressions"))

        # Workload sensitivity
        workload_sensitivity_analyzer = WorkloadSensitivityAnalyzer()
        if not os.path.exists(os.path.join(output_path, "WorkloadSensitivity")):
            os.mkdir(os.path.join(output_path, "WorkloadSensitivity"))
        workload_sensitivity_analyzer.process_data(cs, os.path.join(input_path, case_study),
                                                   os.path.join(output_path, "WorkloadSensitivity"))

        # Workload frequency
        frequency_analyzer = WorkloadFrequencyAnalyzer()
        if not os.path.exists(os.path.join(output_path, "Frequency")):
            os.mkdir(os.path.join(output_path, "Frequency"))
        frequency_analyzer.perform_analysis(os.path.join(output_path, "Frequency"),
                                            os.path.join(input_path, case_study))


if __name__ == "__main__":
    main()
