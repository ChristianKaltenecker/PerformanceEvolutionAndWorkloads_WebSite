import json

from PerformanceEvolution.recall_analyzer import RecallAnalyzer
from analysis_levels import AnalysisLevels
from case_study import CaseStudy
import os
import csv
import sys
from shutil import copyfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import pivot_table
import subprocess
from typing import Dict
from typing import List
from typing import Tuple
from utilities import *
from vif_analysis import VIFAnalyzer
from feature import Feature

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import process_workloads


class OptionLevel(AnalysisLevels):
    name = "OptionLevel"

    SPLConqueror_Path = "/tmp/SPLConqueror/SPLConqueror/CommandLine/bin/Release/CommandLine.exe"

    SBatch_Options = "-p anywhere -A ls-apel --constraint='i5' -n 1 --mem=15000M " \
                     "--time='24:00:00' " \
                     "--output=/scratch/kaltenec/Workloads/slurm_out.log "

    def __init__(self):
        # Initialize for data gathering
        self.changes_in_revisions_and_workloads: Dict[str, Dict[str, List[Tuple[str, bool, str]]]] = dict()
        self.option_changes_for_recall: Dict[str, Dict[str, List[str]]] = dict()
        self.option_changes_for_precision_with_direction: Dict[str, Dict[str, List[Tuple[str, str]]]] = dict()
        self.number_term_changes_per_release: Dict[str, float] = dict()

        self.error_count = 0.0
        self.error_sum = 0

    @staticmethod
    def execute_command(command: str) -> str:
        output = subprocess.getstatusoutput(command)
        status_code = output[0]
        message = output[1]

        # Throw an error if the command was not successfully executed
        if status_code != 0:
            raise RuntimeError(message)

        return message

    def initialize_for_metrics(self, path: str):
        with open(os.path.join(path, "README_post.md"), 'w') as output_file:
            output_file.write("\n")
            output_file.write("| Case Study | Outlier Terms | "
                              "\n")
            output_file.write("| :---: | :---: |\n")

        with open(os.path.join(path, "README_post2.md"), 'w') as output_file:
            # output_file.write("### Error Rates")
            output_file.write("\n")
            output_file.write("| Case Study | Release | Error Rate | "
                              "\n")
            output_file.write("| :---: | :---: | :---: |\n")

    def prepare(self, case_study: CaseStudy, input_path: str) -> None:
        # If no directory 'models' is included, the performance-influence models for each release have to be
        #  created and learned
        input_path = os.path.join(input_path, case_study.name)
        models_path = os.path.join(input_path, 'models')
        workloads = process_workloads.WORKLOADS[str(case_study.name)]
        revisions = list(dict.fromkeys(case_study.configurations.revision))

        if not os.path.exists(models_path):
            print("\n\t\tCreating new slurm job for " + case_study.name + "...", end="")

            super().create_directory(models_path)

            # Copy the FeatureModel.xml
            copyfile(os.path.join(input_path, "FeatureModel.xml"), os.path.join(models_path, "FeatureModel.xml"))

            # Extract the measurement files
            all_configurations = case_study.configurations
            header = list(
                filter(lambda x: x != process_workloads.WORKLOAD_COLUMN_NAME and x != "revision",
                       all_configurations.columns.values))

            self.create_iterative_learning_jobs(all_configurations, header, models_path, workloads, revisions)

        elif not os.path.exists(os.path.join(models_path, "model_base.txt")):
            multicollinearity_features_that_will_be_removed = self.identify_removed_multicollinearity_features(
                case_study)
            # If the iterative models are learned, the results have to be aggregated and new models have to be
            #  learned by using the evaluate-model functionality of SPL Conqueror
            with open(os.path.join(models_path, "learn_opt.txt"), 'w') as general_learn_file:
                terms_across_workloads = dict()
                for workload in workloads:
                    all_terms = dict()
                    for revision in revisions:
                        performance_model, model_error = self.get_performance_model(
                            os.path.join(models_path, f"{workload}_{revision}.log"))
                        if performance_model == "":
                            print(f"Performance model is empty in {case_study.name} {revision} {workload}!")
                            exit(-1)
                        term_dict = self.process_model(performance_model, case_study,
                                                       multicollinearity_features_that_will_be_removed)

                        self.combine_dicts(all_terms, term_dict)

                    self.combine_dicts(terms_across_workloads, all_terms)

                for workload in workloads:
                    terms = self.sort_terms(list(terms_across_workloads.keys()), case_study)

                    self.write_model(terms, os.path.join(models_path, f"model_base_{workload}.txt"))

                    # Optimize the models by using the Variance Influence Factor (VIF)
                    vif_analyzer = VIFAnalyzer(case_study, os.path.join(models_path, f"model_base_{workload}.txt"))
                    term_number = len(vif_analyzer.terms)
                    model_with_countermeasures = vif_analyzer.apply_multicollinearity_countermeasures()
                    if len(model_with_countermeasures) < term_number:
                        print(f"In the case study {case_study.name}, some terms were dropped due to countermeasures.")
                    # Optimize the models for each workload and release
                    for revision in revisions:
                        new_model = vif_analyzer.apply_iterative_vif(model_with_countermeasures, case_study.Performance,
                                                                     os.path.join(models_path,
                                                                                  f"conflicts_{workload}_{revision}.txt"),
                                                                     workload=workload, revision=revision)

                        # Print the new model
                        converted_model = list(map(lambda a: " * ".join(a), new_model))
                        opt_file = os.path.join(models_path, f"model_opt_{workload}_{revision}.txt")
                        self.write_model(converted_model, opt_file)

                    self.create_truemodel_scripts(models_path, revisions, workloads)
                    for revision in revisions:
                        general_learn_file.write(
                            f"mono {self.SPLConqueror_Path} {os.path.join(models_path, f'learn_{workload}_{revision}.a')}\n")

            terms_all_workloads = self.sort_terms(list(terms_across_workloads.keys()), case_study)

            self.write_model(terms_all_workloads, os.path.join(models_path, f"model_base.txt"))

        elif not os.path.exists(os.path.join(models_path, "models.csv")):
            self.extract_models(models_path, revisions, workloads, "opt")

    def create_truemodel_scripts(self, models_path: str, revisions: List[str], workloads: List[str]) -> None:
        # Learn the model
        # Create automation script for SPL Conqueror
        # Set up slurm jobs
        for workload in workloads:
            with open(os.path.join(models_path, f'jobs_{workload}.txt'), 'w', newline="\n") as job_file:
                for revision in revisions:
                    # job_string = "export LD_LIBRARY_PATH=/scratch/kaltenec/lib:$LD_LIBRARY_PATH && "
                    job_string = f"mono {self.SPLConqueror_Path} {os.path.join(models_path, f'learn_{workload}_{revision}.a')}"
                    job_file.write(job_string + "\n")
                    with open(os.path.join(models_path, f'learn_{workload}_{revision}.a'), "w",
                              newline="\n") as a_file:
                        all_lines = list()
                        # log
                        all_lines.append(f"log ./{workload}_{str(revision)}_opt.log")
                        # VM
                        all_lines.append('vm ./FeatureModel.xml')
                        # Measurements
                        all_lines.append(f'all ./{workload}_{str(revision)}.csv')
                        all_lines.append('nfp performance')
                        # truemodel
                        all_lines.append('select-all-measurements true')
                        all_lines.append(f'truemodel model_opt_{workload}_{revision}.txt')
                        a_file.writelines(list(map(lambda x: x + "\n", all_lines)))

    def extract_models(self, models_path: str, revisions: List[str], workloads: List[str], suffix: str) -> None:
        # (only if all performance-influence models are learned:) Combine all performance-influence models
        #  into one single file for each release
        # Use the model_opt.txt file as header
        with open(os.path.join(models_path, "model_base.txt"), 'r') as model_file:
            header = model_file.readlines()
            header = list(map(lambda x: x.replace("\n", ""), header))
            header.insert(0, "revision")
            header.insert(0, "workload")
            header.insert(len(header), "error")
        with open(os.path.join(models_path, "models.csv"), 'w', newline="\n") as models_file:
            dict_writer = csv.DictWriter(models_file, delimiter=";", fieldnames=header)
            dict_writer.writeheader()
            for workload in workloads:
                for revision in revisions:
                    revision_dict = dict()
                    revision_dict["workload"] = f"{workload}"
                    revision_dict["revision"] = f"{revision}"
                    performance_model, model_error = self.get_performance_model(
                        os.path.join(models_path, f"{workload}_{str(revision)}_opt.log"))
                    terms = performance_model.split("+")
                    for term in terms:
                        elements = term.strip().split("*")
                        elements = list(map(lambda x: x.strip(), elements))
                        options = ' * '.join(elements[1:])
                        revision_dict[options] = elements[0]
                    revision_dict["error"] = model_error
                    dict_writer.writerow(revision_dict)

    def sort_terms(self, term_list: List[str], case_study: CaseStudy) -> List[str]:
        """
        Sort the terms in the given list.
        The first element is the mandatory feature. Afterwards, the higher interactions follow.
        The last elements are the lower interactions and individual features.
        """
        list_to_sort: List[List[str]] = []
        max_terms = 1
        for term in term_list:
            elements = term.split('*')
            if len(elements) > max_terms:
                max_terms = len(elements)
            list_to_sort.append(elements)

        def term_order(elem: List[str]) -> int:
            if len(elem) == 1 and case_study.is_strictly_mandatory(elem[0]):
                return -sys.maxsize - 1
            else:
                return len(elem) - max_terms - 1

        list_to_sort.sort(key=term_order)
        list_to_return: List[str] = []
        for term in list_to_sort:
            list_to_return.append(" * ".join(term))
        return list_to_return

    def create_iterative_learning_jobs(self, all_configurations: pd.DataFrame, header: List[str], models_path: str,
                                       workloads: List[str], revisions: List[str]):
        # Set up slurm jobs
        with open(os.path.join(models_path, 'jobs.txt'), 'w', newline="\n") as job_file:
            for revision in revisions:
                for workload in workloads:
                    # job_string = "export LD_LIBRARY_PATH=/scratch/kaltenec/lib:$LD_LIBRARY_PATH && "
                    job_string = "mono " + self.SPLConqueror_Path + " " + os.path.join(models_path,
                                                                                       f"learn_{workload}_{revision}.a")
                    job_file.write(job_string + "\n")

                    # Create measurements file
                    with open(os.path.join(models_path, f"{workload}_{revision}.csv"), "w", newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, delimiter=';', fieldnames=header)
                        writer.writeheader()
                        configurations_workload = all_configurations[
                            np.logical_and(np.logical_and(all_configurations['revision'] == revision,
                                                          all_configurations[
                                                              process_workloads.WORKLOAD_COLUMN_NAME] == workload),
                                           pd.to_numeric(all_configurations['performance']) != 0.0)]
                        if len(configurations_workload) == 0:
                            print(f"Workload {workload} of release {revision} has no configurations != 0")
                        for index, configuration in configurations_workload.iterrows():
                            configuration_dict = {}
                            for column in header:
                                configuration_dict[column] = configuration[column]
                            writer.writerow(configuration_dict)
                    # Create automation script for SPL Conqueror
                    with open(os.path.join(models_path, f"learn_{workload}_{revision}.a"), "w", newline="\n") as a_file:
                        all_lines = list()
                        # log
                        all_lines.append(f"log ./{workload}_{revision}.log")
                        # ML-settings
                        all_lines.append(
                            'mlsettings epsilon:0 lossFunction:RELATIVE parallelization:True bagging:False '
                            'considerEpsilonTube:False useBackward:False abortError:5 '
                            'limitFeatureSize:False quadraticFunctionSupport:False crossValidation:False '
                            'learn-logFunction:False learn-accumulatedLogFunction:False '
                            'learn-asymFunction:False learn-ratioFunction:False numberOfRounds:70 '
                            'backwardErrorDelta:1 minImprovementPerRound:0.1 withHierarchy:False')
                        # VM
                        all_lines.append('vm ./FeatureModel.xml')
                        # Measurements
                        all_lines.append(f"all ./{workload}_{revision}.csv")
                        all_lines.append('select-all-measurements true')
                        all_lines.append('nfp performance')
                        # learn-splconqueror
                        all_lines.append('learn-splconqueror')
                        all_lines.append('clean-global')
                        a_file.writelines(list(map(lambda x: x + "\n", all_lines)))

    @staticmethod
    def get_performance_model(path: str) -> Tuple[str, float]:
        model = ""
        error = 0.0
        with open(path, "r") as log_file:
            for line in log_file.readlines():
                if ";" in line:
                    elements = line.split(";")
                    model = elements[1]
                    error = float(elements[2])
            return model, error

    def process_model(self, model: str, case_study: CaseStudy,
                      multicollinearity_features_that_will_be_removed: Dict[Feature, str]) -> Dict:
        terms = model.split("+")
        term_dict = dict()
        for term in terms:
            features_in_term = term.split("*")[1:]
            term_list = [[]]
            for feature_in_term in features_in_term:
                # If alternative feature
                feature_in_term = feature_in_term.strip()
                exclusions = case_study.features[feature_in_term].exclusions
                if case_study.features[feature_in_term] in multicollinearity_features_that_will_be_removed:
                    to_replace_by = [
                        multicollinearity_features_that_will_be_removed[case_study.features[feature_in_term]]]
                    for exclusion in exclusions:
                        to_replace_by.append(exclusion)
                    term_list = self.multiply_lists(term_list, to_replace_by)
                else:
                    term_list = self.multiply_lists(term_list, [feature_in_term])
            for term_element in term_list:
                term_dict[' * '.join(term_element)] = 0
        return term_dict

    @staticmethod
    def multiply_lists(first_list: List, second_list: List) -> List:
        result = list()
        for y in second_list:
            for x in first_list:
                l = list(x)
                l.append(y)
                result.append(sorted(l))
        return result

    @staticmethod
    def combine_dicts(first_dict: Dict, second_dict: Dict) -> None:
        for key in second_dict.keys():
            first_dict[key] = second_dict[key]

    @staticmethod
    def write_model(terms: List[str], path: str) -> None:
        with open(path, 'w') as model_file:
            for term in terms:
                term = term.replace("  ", " ")
                model_file.write(term + "\n")

    def evaluate_metrics(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        pass

    def generate_plots(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        if os.path.exists(os.path.join(input_path, case_study.name, "models", "models.csv")):
            with open(os.path.join(input_path, case_study.name, "models", "models.csv")) as models_file:
                self.generate_influence_difference_plots(case_study, input_path, models_file, path)

        if os.path.exists(os.path.join(input_path, case_study.name, "models", "models.csv")):
            with open(os.path.join(input_path, case_study.name, "models", "models.csv")) as models_file:
                # (I) Prepare the data for the changes
                performance_models = pd.read_csv(models_file, sep=";")
                workloads = process_workloads.WORKLOADS[str(case_study.name)]
                revisions = list(dict.fromkeys(case_study.configurations.revision))
                for workload in workloads:
                    workload_models = performance_models[performance_models['workload'] == workload].copy()
                    workload_models.dropna(how='all', axis=1, inplace=True)
                    plot_data = np.zeros((len(revisions), len(performance_models.columns) - 3))
                    for y in range(0, len(revisions)):
                        plot_data[y] = performance_models.iloc[y][2:len(
                            performance_models.columns) - 1]

                    config_data = case_study.configurations[
                        case_study.configurations[process_workloads.WORKLOAD_COLUMN_NAME] == workload]
                    mean_values = pivot_table(config_data[config_data['performance'] != 1800], values='performance',
                                              index=['revision'])
                    mean_values = mean_values.iloc[mean_values.index.map(revisions.index).argsort()]

                    deviation_values = pivot_table(case_study.deviations, values='performance',
                                                   index=['revision'])
                    deviation_values = deviation_values.iloc[deviation_values.index.map(revisions.index).argsort()]
                    deviation_values.reset_index(inplace=True)
                    changed = np.zeros(len(revisions) - 1)
                    relevant_performance_model_columns = dict()

                    for y in range(1, len(revisions)):
                        relevant_performance_model_columns[y - 1] = list()
                        standard_deviation = max(mean_values.iloc[len(revisions) - 1 - y]['performance'] *
                                                 deviation_values.iloc[len(revisions) - 1 - y]['performance'],
                                                 mean_values.iloc[len(revisions) - y]['performance'] *
                                                 deviation_values.iloc[len(revisions) - y]['performance'])
                        min_value = 2 * standard_deviation
                        for i in range(0, len(performance_models.columns) - 3):
                            difference = plot_data[y - 1][i] - plot_data[y][i]
                            if abs(difference) > min_value:
                                changed[y - 1] += 1
                                relevant_performance_model_columns[y - 1].append(i)

                        release_tag = f"{revisions[y - 1]} - {revisions[y]}"
                        if release_tag not in self.number_term_changes_per_release:
                            self.number_term_changes_per_release[release_tag] = 0
                        workload_model_for_release = workload_models[
                            workload_models['revision'] == revisions[y - 1]].copy()
                        workload_model_for_release.dropna(how='all', axis=1, inplace=True)
                        self.number_term_changes_per_release[release_tag] += (
                                float(changed[y - 1]) / (len(workload_model_for_release.columns) - 2) * 100)

                        changed[y - 1] = float(changed[y - 1]) / (len(performance_models.columns) - 2) * 100

                for releases in self.number_term_changes_per_release.keys():
                    self.number_term_changes_per_release[releases] /= len(workloads)
                self.generate_barplots_per_release(case_study, path)

    def generate_barplots_per_release(self, case_study: CaseStudy, output_path: str) -> None:
        term_changes = list()
        for releases in sorted(self.number_term_changes_per_release.keys()):
            term_changes.append(self.number_term_changes_per_release[releases])

        plt.rcParams['xtick.bottom'] = True
        plt.figure(figsize=(12, 11))
        sns.set_color_codes("muted")
        ax = sns.barplot(x=[label.replace("_", ".") for label in sorted(self.number_term_changes_per_release.keys())],
                         y=term_changes,
                         color='b')

        fig = ax.get_figure()
        ax.set_ylabel("Terms [%]", fontsize=50)
        ax.set_xlabel("Releases", fontsize=50, labelpad=20)
        plt.xticks(rotation=45, ha='right')
        # ax.tick_params(labelsize=50)
        ax.set_ylim([0, 100])

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, 'termChanges.pdf'))
        plt.close(fig)

    def store_performance_change(self, term: str, from_revision: str, to_revision: str, workload: str, speed_up: bool,
                                 amount_change: str, renamed_term: str = None) -> None:
        revisions = f"{from_revision} - {to_revision}"
        if revisions not in self.option_changes_for_recall:
            self.option_changes_for_recall[revisions] = dict()
            self.option_changes_for_precision_with_direction[revisions] = dict()
        if workload not in self.option_changes_for_recall[revisions]:
            self.option_changes_for_recall[revisions][workload] = list()
            self.option_changes_for_precision_with_direction[revisions][workload] = list()
        if term not in self.option_changes_for_recall[revisions][workload]:
            self.option_changes_for_recall[revisions][workload].append(term)
            self.option_changes_for_precision_with_direction[revisions][workload].append((term, str(speed_up)))

        if renamed_term is not None:
            term = renamed_term
        if term not in self.changes_in_revisions_and_workloads:
            self.changes_in_revisions_and_workloads[term] = dict()
        if revisions not in self.changes_in_revisions_and_workloads[term]:
            self.changes_in_revisions_and_workloads[term][revisions] = list()
        self.changes_in_revisions_and_workloads[term][revisions].append((workload, speed_up, amount_change))

    def generate_influence_difference_plots(self, case_study, input_path, models_file, path):
        # (I) Plot the coefficients of the terms with a heatmap
        performance_models = pd.read_csv(models_file, sep=";")
        revisions = list(dict.fromkeys(case_study.configurations.revision))
        workloads = process_workloads.WORKLOADS[str(case_study.name)]
        colums_to_add, columns_to_add_from = self.columns_to_add_for_multicollinearity(case_study,
                                                                                       list(performance_models.columns)[
                                                                                       2:])
        for workload in workloads:
            plot_data = np.zeros((len(revisions), len(performance_models.columns) - 3))
            performance_models_workload = performance_models[
                performance_models[process_workloads.WORKLOAD_COLUMN_NAME] == workload]
            for y in range(0, len(revisions)):
                plot_data[len(revisions) - 1 - y] = performance_models_workload.iloc[y][2:len(
                    performance_models_workload.columns) - 1]

            for y in range(0, len(revisions)):
                for column_to_add_index, columns_to_add_to_indexes in colums_to_add:
                    for colum_to_add_to_index in columns_to_add_to_indexes:
                        plot_data[len(revisions) - 1 - y][colum_to_add_to_index] += plot_data[len(revisions) - 1 - y][
                            column_to_add_index]
            for y in range(0, len(revisions)):
                for column_to_add_index, columns_to_add_to_indexes in columns_to_add_from:
                    for colum_to_add_from_index in columns_to_add_to_indexes:
                        plot_data[len(revisions) - 1 - y][
                            column_to_add_index] += performance_models_workload.iloc[y][colum_to_add_from_index + 2]

            # Export the processed data for the recall analysis
            self.write_plot_data(plot_data, os.path.join(input_path, f"plot_data_{workload}.json"))

            fontsize = 20
            cmap = plt.get_cmap('RdBu_r')
            fig = plt.figure(figsize=(18, 10))
            ax = fig.add_subplot(1, 1, 1)
            greatest_value = max(abs(np.nanmin(plot_data)), abs(np.nanmax(plot_data)))
            cm = ax.pcolormesh(plot_data, cmap=cmap, vmin=-greatest_value, vmax=greatest_value)
            ax.set_title(case_study.name, fontsize=fontsize)
            ax.set_ylabel('Release', fontsize=fontsize)
            ax.set_xlabel('Term', fontsize=fontsize)
            ax.set_yticks(np.arange(0.5, len(revisions) + 0.5, step=1.0))
            ax.set_yticklabels(reversed(revisions))
            tmp = list(performance_models_workload.columns[2:-1])
            ax.set_xticks(np.arange(0.5, len(tmp) + 0.5, step=1.0))
            ax.set_xticklabels(tmp, rotation=45, ha='right', fontsize=20)
            cb = fig.colorbar(cm, ax=ax)
            cb.ax.tick_params(labelsize=fontsize)
            cb.set_label('Influence [s]', fontsize=fontsize)
            fig = ax.get_figure()
            fig.tight_layout()
            super().create_directory(os.path.join(path, workload, 'AbsoluteInfluence'))
            fig.savefig(os.path.join(path, workload, 'AbsoluteInfluence', 'configurationsInfluence.pdf'))
            plt.close(fig)
            # (II) Plot the differences of the coefficients of the terms with a heatmap
            plot_data2 = np.copy(plot_data)
            config_data = case_study.configurations[
                case_study.configurations[process_workloads.WORKLOAD_COLUMN_NAME] == workload]
            mean_values = pivot_table(config_data[config_data['performance'] != 1800], values='performance',
                                      index=['revision'])
            mean_values = mean_values.iloc[mean_values.index.map(revisions.index).argsort()]
            term_dictionary = dict()
            term_ci_dictionary = dict()
            deviation_values = pivot_table(case_study.deviations, values='performance', index=['revision'])
            deviation_values = deviation_values.iloc[deviation_values.index.map(revisions.index).argsort()]
            deviation_values.reset_index(inplace=True)
            term_renaming = self.column_renaming_for_multicollinearity(case_study)
            with open(os.path.join(input_path, case_study.name, "relevantTerms.txt"), 'w') as term_file:
                for y in range(1, len(revisions)):
                    terms = ""

                    term_dictionary[revisions[y - 1] + "-" + revisions[y]] = [[], 0, [0, 0, 0, 0, 0, 0, 0], 0]
                    term_ci_dictionary[revisions[y - 1] + "-" + revisions[y]] = [[], 0]
                    standard_deviation = max(mean_values.iloc[len(revisions) - 1 - y]['performance'] * \
                                             deviation_values.iloc[len(revisions) - 1 - y]['performance'],
                                             mean_values.iloc[len(revisions) - y]['performance'] * \
                                             deviation_values.iloc[len(revisions) - y]['performance']) / \
                                         case_study.get_division_factor()
                    min_value = 2 * standard_deviation
                    relevant_column_counter = 0
                    for i in range(0, len(performance_models_workload.columns) - 3):
                        difference = - (plot_data[len(revisions) - y][i] - plot_data[len(revisions) - 1 - y][i])
                        term_dictionary[revisions[y - 1] + "-" + revisions[y]][3] += 1
                        if abs(difference) > min_value and plot_data[len(revisions) - 1 - y][i] != 0:
                            relevant_column_counter += 1
                            plot_data2[len(revisions) - 1 - y][i] = difference
                            term_dictionary[revisions[y - 1] + "-" + revisions[y]][0].append(
                                performance_models_workload.columns[i + 1])
                            term_dictionary[revisions[y - 1] + "-" + revisions[y]][1] += difference

                            if i == 0:
                                terms += "root; "
                                term_dictionary[revisions[y - 1] + "-" + revisions[y]][2][0] += 1
                            else:
                                terms += performance_models_workload.columns[i + 1] + ";"
                                term_dictionary[revisions[y - 1] + "-" + revisions[y]][2][
                                    len(performance_models_workload.columns[i + 1].split('*'))] += 1

                            # Collect the change in the form of: term -> revision -> (workload, speed up -- boolean)
                            change = difference / ((mean_values.iloc[len(revisions) - y - 1]['performance'] +
                                                    mean_values.iloc[len(revisions) - y]['performance']) / 2) * 100
                            # if abs(change) > 1:
                            if performance_models_workload.columns[i + 2] in term_renaming:
                                self.store_performance_change(performance_models_workload.columns[i + 2],
                                                              revisions[y - 1],
                                                              revisions[y], workload, difference < 0,
                                                              "{:.2f}%".format(change),
                                                              renamed_term=term_renaming[
                                                                  performance_models_workload.columns[i + 2]])
                            else:
                                self.store_performance_change(performance_models_workload.columns[i + 2],
                                                              revisions[y - 1],
                                                              revisions[y], workload, difference < 0,
                                                              "{:.2f}%".format(change))
                        else:
                            plot_data2[len(revisions) - 1 - y][i] = 0

                        if plot_data[len(revisions) - y - 1][i] != 0 and plot_data[len(revisions) - y][i] == 0:
                            print(
                                f"Newer release results in timeout: {revisions[y - 1]} - {revisions[y]} Workload:{workload} Term: {performance_models_workload.columns[i + 2]}")

                        if confidence_interval_disjoint(case_study.name,
                                                        plot_data[len(revisions) - 1 - y][i],
                                                        deviation_values.iloc[len(revisions) - 1 - y][
                                                            'performance'],
                                                        plot_data[len(revisions) - y][i],
                                                        deviation_values.iloc[len(revisions) - y]['performance']):
                            term_ci_dictionary[revisions[y - 1] + "-" + revisions[y]][0].append(
                                performance_models_workload.columns[i])
                            term_ci_dictionary[revisions[y - 1] + "-" + revisions[y]][1] += difference

                    # self.changes_distribution[
                    #    int(relevant_column_counter * 100 / (len(performance_models.columns) - 2.0))] += 1

                    if terms != "" and terms != performance_models_workload.columns[1]:
                        term_file.write(f"{workload} -- {revisions[y]}: {terms}\n")

                    term_dictionary[revisions[y - 1] + "-" + revisions[y]][1] /= standard_deviation
            plot_data2 = np.delete(plot_data2, len(revisions) - 1, axis=0)
            cmap = plt.get_cmap('RdBu_r')
            fig = plt.figure(figsize=(18, 8))
            ax = fig.add_subplot(1, 1, 1)
            greatest_value = max(abs(np.nanmin(plot_data2)), abs(np.nanmax(plot_data2)))
            # The following code is only needed for the pervolution renewal
            cm = ax.pcolormesh(plot_data2, cmap=cmap, vmin=-greatest_value, vmax=greatest_value)
            # ax.set_title(case_study.name, fontsize=fontsize)
            ax.set_ylabel('Release', fontsize=fontsize)
            ax.set_xlabel('Configuration Choice', fontsize=fontsize)
            ax.set_yticks(range(0, len(revisions)))
            ax.set_yticklabels(reversed(revisions), fontsize=20)
            tmp = list(performance_models_workload.columns[2:-1])
            tmp[0] = 'root * blind'
            ax.set_xticks(np.arange(0.5, len(tmp) + 0.5, step=1.0))
            ax.set_xticklabels(tmp, rotation=45, ha='right', fontsize=20)
            cb = fig.colorbar(cm, ax=ax, extend='both')
            cb.ax.tick_params(labelsize=fontsize)
            cb.set_label('Influence Difference [s]', fontsize=fontsize)
            fig = ax.get_figure()
            fig.tight_layout()
            self.create_directory(os.path.join(path, workload, 'InfluenceDifference'))
            fig.savefig(os.path.join(path, workload, 'InfluenceDifference', 'influenceDifference.pdf'))
            plt.close(fig)

    def write_plot_data(self, matrix: np.array, path: str) -> None:
        matrix.dump(path)

    def replace_terms(self, terms):
        term_replacement_dict = {'root': 'Root', 'lzo': 'LZO', 'auth_sha512': 'SHA512',
                                 'auth_sha1': 'SHA1', 'auth_rsa_sha512': 'RSA SHA512',
                                 'prng_sha512': 'SHA512 PRN Gen.',
                                 'prng_rsa_sha512': 'SHA512 RSA PRN Gen.',
                                 'prng_sha1': 'SHA1 PRN Gen.', 'TCP_NODEAL': 'TCP No Delay',
                                 'AES_128_CBC': 'AES-128-CBC'}
        new_terms = []
        for term in terms:
            configuration_options = term.split('*')
            new_configuration_options = []
            for configuration_option in configuration_options:
                new_configuration_options.append(term_replacement_dict[configuration_option.strip()])
            new_terms.append(' · '.join(new_configuration_options))

        return new_terms

    def columns_to_add_for_multicollinearity(self, case_study: CaseStudy, columns: List[str]) -> Tuple[List[
        Tuple[int, List[int]]], List[Tuple[int, List[int]]]]:
        removed_multicollinear_features = self.identify_removed_multicollinearity_features(case_study)

        columns_to_add: List[Tuple[int, List[int]]] = list()
        features_to_investigate = case_study.features['root'].children
        # It is crucial to preserve the order from the feature model; otherwise, the columns will be added wrongly
        while len(features_to_investigate) > 0:
            current_feature: Feature = case_study.features[features_to_investigate[0]]
            features_to_investigate = features_to_investigate[1:]
            if len(current_feature.children) > 0:
                for child in current_feature.children:
                    features_to_investigate.append(child)
            if current_feature in removed_multicollinear_features:
                # Find transitive parents if the parent is a removed multicollinear feature
                current_replacement = case_study.features[removed_multicollinear_features[current_feature]]
                while current_replacement in removed_multicollinear_features and removed_multicollinear_features[
                    current_replacement] != "root":
                    current_replacement = case_study.features[removed_multicollinear_features[current_replacement]]
                if current_replacement.name != removed_multicollinear_features[current_feature]:
                    removed_multicollinear_features[current_feature] = current_replacement.name
            if current_feature not in removed_multicollinear_features or removed_multicollinear_features[
                current_feature] not in columns:
                continue
            parent_pos = columns.index(removed_multicollinear_features[current_feature])
            alternative_pos_list: List[int] = list()
            if len(current_feature.alternatives) != 0:
                for alternative in current_feature.alternatives:
                    alternative_post = columns.index(alternative)
                    alternative_pos_list.append(alternative_post)
                columns_to_add.append((parent_pos, alternative_pos_list))
            elif not current_feature.mandatory:
                columns_to_add.append((parent_pos, [columns.index(current_feature.name)]))

        # Handle interactions by adding to them the base influence and the influence of parent features
        columns_to_add_from: List[Tuple[int, List[int]]] = list()
        for column in columns:
            terms = column.split(' * ')
            if len(terms) == 1:
                continue

            # Add the base term. Note that there is only one base term.
            for column_to_add, column_list in columns_to_add:
                if columns.index(terms[0]) in column_list:
                    column_list.append(columns.index(column))
                    break

            # Add the feature terms afterwards
            feature_columns: List[int] = list()
            for term in terms:
                feature_columns.append(columns.index(term.strip()))
            columns_to_add_from.append((columns.index(column), feature_columns))

        return columns_to_add, columns_to_add_from

    def identify_removed_multicollinearity_features(self, case_study: CaseStudy) -> Dict[Feature, str]:
        # Identify all alternative features and take the first one, since it was removed for multicollinearity
        removed_multicollinear_features: Dict[Feature, str] = dict()
        for feature_name in case_study.features:
            feature = case_study.features[feature_name]
            if len(feature.alternatives) > 0 and all(
                    case_study.features[alternative] not in removed_multicollinear_features for alternative in
                    feature.alternatives):
                removed_multicollinear_features[feature] = ""
            elif not feature.mandatory:
                removed_multicollinear_features[feature] = ""
        # Identify the parent feature (transitive) that is mandatory and not root
        # This is the feature that is presumably used in MLR and whose influence has to be added to the other
        # alternatives.
        for feature in removed_multicollinear_features:
            removed_multicollinear_features[feature] = self.find_suitable_parent(case_study, feature)
        return removed_multicollinear_features

    def find_suitable_parent(self, case_study: CaseStudy, feature: Feature) -> str:
        parent = feature.parent
        while case_study.features[parent].parent != "root" and case_study.features[parent].mandatory and len(
                case_study.features[parent].alternatives) == 0:
            parent = case_study.features[parent].parent
        return parent

    def column_renaming_for_multicollinearity(self, case_study: CaseStudy) -> Dict[str, str]:
        removed_multicollinear_features = self.identify_removed_multicollinearity_features(case_study)

        column_renaming: Dict[str, str] = dict()
        for feature in removed_multicollinear_features:
            column_renaming[removed_multicollinear_features[
                feature]] = f"{removed_multicollinear_features[feature]} ( * {feature.name})"

        return column_renaming

    def write_error_rates(self, output_file, case_study_name, performance_models):
        """
            Writes the error rates of the case study in a README file
            :param output_file:
            :param performance_models:
            :return:
            """
        for i in range(0, len(performance_models)):
            workload = performance_models[process_workloads.WORKLOAD_COLUMN_NAME][i]
            error = performance_models['error'][i]
            if i == 0:
                output_file.write("| " + case_study_name + " | ")
            else:
                output_file.write("| | ")
            output_file.write(str(workload) + " | " + "{:.2f}".format(error) + "% |\n")
            self.error_sum += float(error)
            self.error_count += 1

    def write_frequency(self, output_file, workload_frequency):
        workload_frequency_ranking = list(workload_frequency.keys())
        workload_frequency_ranking = list(filter(lambda z: workload_frequency[z][1] != 0, workload_frequency_ranking))
        workload_frequency_ranking.sort(key=lambda z: workload_frequency[z][1], reverse=True)
        for workload in workload_frequency_ranking:
            output_file.write(workload + ": ")
            # Write all terms
            output_file.write(str(len(workload_frequency[workload][0])) + " ")
            output_file.write(str(workload_frequency[workload][2]) + " -- " +
                              "{0:.2f}".format(
                                  sum(workload_frequency[workload][2]) / workload_frequency[workload][3] * 100) +
                              "%; ")

    def finish(self, path: str, input_path: str) -> None:
        # print("Overall average error:" + "{:.2f}".format(self.error_sum / self.error_count))

        print(f"File: {os.path.join(path, 'identified_changes.md')}")
        with open(os.path.join(path, "identified_changes.md"), 'w') as change_file:
            change_file.write("| Term | Releases | Workloads | "
                              "\n")
            change_file.write("| :---: | :---: | :---: |\n")
            for term in sorted(self.changes_in_revisions_and_workloads):
                for release in sorted(self.changes_in_revisions_and_workloads[term].keys()):
                    change_file.write(f"| {term} | {release} | ")
                    for (workload, speed_up, change) in self.changes_in_revisions_and_workloads[term][release]:
                        if speed_up:
                            change_file.write(f"{workload}({change}↑) ")
                        else:
                            change_file.write(f"{workload}({change}↓) ")
                    change_file.write("|\n")
        with open(os.path.join(input_path, 'changed_options.json'), 'w') as changed_options:
            changed_options.write(json.dumps(self.option_changes_for_recall))
        with open(os.path.join(input_path, 'changed_options_with_direction.json'), 'w') as changed_options:
            changed_options.write(json.dumps(self.option_changes_for_precision_with_direction))
