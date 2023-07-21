import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from PerformanceEvolution.case_study import CaseStudy


class RecallAnalyzer:

    def __init__(self):
        self.configurations = None
        self.options = None
        self.options_with_directions = None
        self.configurations_with_directions = None
        self.affected_configurations_per_term_and_release: Dict[str, Dict[str, Dict[str, float]]] = dict()

    def perform_analysis(self, path: str, case_study: CaseStudy, models_path: str, input_path: str) -> None:
        with open(os.path.join(input_path, 'changed_configurations.json'), 'r') as changed_configurations:
            self.configurations = json.load(changed_configurations)
        with open(os.path.join(input_path, 'changed_options.json'), 'r') as changed_options:
            self.options = json.load(changed_options)
        with open(os.path.join(input_path, 'changed_configurations_with_direction.json'), 'r') as changed_configurations:
            self.configurations_with_directions = json.load(changed_configurations)
        with open(os.path.join(input_path, 'changed_options_with_direction.json'), 'r') as changed_options:
            self.options_with_directions = json.load(changed_options)

        performance_models = pd.read_csv(models_path, sep=";")
        performance_models_with_error = pd.read_csv(models_path, sep=";")
        performance_models.drop(columns=['error'], inplace=True)
        performance_models.replace("", float("NaN"), inplace=True)

        all_revision = list(case_study.configurations["revision"].unique())

        # First, determine the affected configurations per term
        for revisions in self.configurations:
            first_release = revisions.split(' - ')[0]
            self.affected_configurations_per_term_and_release[revisions] = dict()
            for workload in self.configurations[revisions]:
                self.affected_configurations_per_term_and_release[revisions][workload] = dict()
                current_performance_models = performance_models.loc[
                    (performance_models['revision'] == first_release) & (
                            performance_models['workload'] == workload)].copy()
                # Remove all columns that are empty
                current_performance_models.dropna(how='all', axis=1, inplace=True)

                # Iterate over all columns (except for the first one) and enter them in the according dictionary
                for term in current_performance_models.columns[2:]:
                    affected_configurations = case_study.configurations.loc[
                        (case_study.configurations['revision'] == first_release) & (
                                case_study.configurations['workload'] == workload)]
                    options = term.split(' * ')
                    for option in options:
                        affected_configurations = affected_configurations[affected_configurations[option] == '1']
                    number_configurations = len(affected_configurations)
                    self.affected_configurations_per_term_and_release[revisions][workload][term] = number_configurations

        # After determining that, we can search for the most specific term corresponding to a configuration
        total_changed_configurations = 0
        total_confirmed_changes = 0
        total_confirmed_changes_with_direction = 0
        confirmed_changes_per_workload = dict()
        total_changes_per_workload = dict()
        relevant_change = 0
        relevant_error = 0
        for revisions in self.configurations:
            rev = revisions.split(" - ")
            # These indexes are for the option_infos variable
            first_revision_row = len(all_revision) - 1 - all_revision.index(rev[0])
            second_revision_row = len(all_revision) - 1 - all_revision.index(rev[1])

            # Consider direction
            for workload in self.configurations_with_directions[revisions]:
                for configuration, diff in self.configurations_with_directions[revisions][workload]:
                    affected_terms = self.find_affected_terms(revisions, configuration, workload)
                    for affected_term in affected_terms:
                        if workload not in self.options[revisions]:
                            continue
                        found = False
                        for term, speed_up in self.options_with_directions[revisions][workload]:
                            if term == affected_term:
                                if speed_up == "True" and float(diff) < 0:
                                    total_confirmed_changes_with_direction += 1
                                    found = True
                                elif speed_up == "False" and float(diff) > 0:
                                    total_confirmed_changes_with_direction += 1
                                    found = True
                                break
                        if found:
                            break

            # Ignore direction
            for workload in self.configurations[revisions]:
                option_infos = np.load(os.path.join(input_path, "..", f"plot_data_{workload}.json"), allow_pickle=True)

                if workload not in confirmed_changes_per_workload:
                    confirmed_changes_per_workload[workload] = 0
                    total_changes_per_workload[workload] = 0

                total_changed_configurations += len(self.configurations[revisions][workload])
                for configuration in self.configurations[revisions][workload]:
                    total_changes_per_workload[workload] += 1
                    affected_terms = self.find_affected_terms(revisions, configuration, workload)

                    found = False
                    for affected_term in affected_terms:
                        if workload in self.options[revisions] and affected_term in self.options[revisions][workload]:
                            confirmed_changes_per_workload[workload] += 1
                            total_confirmed_changes += 1
                            found = True
                            break
                    if not found:
                        for affected_term in affected_terms:
                            index_of_affected_term = list(performance_models.columns).index(affected_term) - 2
                            if abs(option_infos[first_revision_row][index_of_affected_term] -
                                   option_infos[second_revision_row][
                                       index_of_affected_term]) > self.get_deviation_of_configuration(case_study,
                                                                                                      configuration,
                                                                                                      revisions,
                                                                                                      workload):
                                relevant_change += 1
                                break
                            elif max(float(performance_models_with_error.loc[
                                               (performance_models_with_error['revision'] == rev[0]) & (
                                                       performance_models_with_error['workload'] == workload)][
                                               "error"].iloc[0]),
                                     float(performance_models_with_error.loc[
                                               (performance_models_with_error['revision'] == rev[1]) & (
                                                       performance_models_with_error['workload'] == workload)][
                                               "error"].iloc[0])) > 10:
                                relevant_error += 1

        self.create_latex_table_for_recall_per_workload(confirmed_changes_per_workload, total_changes_per_workload,
                                                        path)
        # Recall:
        recall = (total_confirmed_changes * 1.0) / (total_changed_configurations * 1.0) * 100.0
        print(f"Total number of configration level changes: {total_changed_configurations}")
        print(f"Recall: {recall}%")
        print(f"Affected by different metrics: {(relevant_change * 1.0) / total_changed_configurations * 100.0}")
        print(f"Affected by high error: {(relevant_error * 1.0) / total_changed_configurations * 100.0}")
        print(f"Recall with direction: {(total_confirmed_changes_with_direction * 1.0) / total_changed_configurations * 100.0}%")

    def create_latex_table_for_recall_per_workload(self, confirmed_changes: Dict[str, int],
                                                   total_changes: Dict[str, int], path: str) -> None:
        workloads = list()
        recall_values = list()
        for workload in sorted(confirmed_changes.keys()):
            workloads.append(workload.replace("_", "\\_"))
            recall_values.append(float(confirmed_changes[workload]) / float(total_changes[workload]) * 100.0)
        df = pd.DataFrame(dict(Workload=workloads, Recall=recall_values))

        # Retrieve standard deviation for dissertation
        print(f"Standard deviation of recall: {np.std(df['Recall'])}%")

        fig = plt.figure(figsize=(5, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim([0, 100])
        sns.violinplot(data=df, y="Recall", linewidth=3, cut=0)
        ax.set_ylabel("Recall [%]")
        fig.tight_layout()
        fig.savefig(os.path.join(path, 'recall_violin.pdf'))
        plt.close(fig)

        # Convert the changes in the configuration level
        with open(os.path.join(path, 'recall_per_workload.tex'), 'w') as tex_file:
            tex_file.write(df.to_latex(index=False, float_format="${:.2f}\\%$".format))

    def get_deviation_of_configuration(self, case_study: CaseStudy, configuration: str, releases: str,
                                       workload: str) -> float:
        configuration_options = configuration.split(" ")
        split_releases = releases.split(" - ")
        workload_deviations = case_study.deviations[case_study.deviations["workload"] == workload]
        workload_performance = case_study.configurations[case_study.configurations["workload"] == workload]
        workload_deviations = workload_deviations.loc[
            (workload_deviations["revision"] == split_releases[0]) | (
                    workload_deviations["revision"] == split_releases[1])]
        workload_performance = workload_performance.loc[
            (workload_performance["revision"] == split_releases[0]) | (
                    workload_performance["revision"] == split_releases[1])]
        for option in workload_deviations.columns:
            if option == "revision":
                break
            if option in configuration_options or option == "root":
                workload_deviations = workload_deviations[workload_deviations[option] == "1"]
                workload_performance = workload_performance[workload_performance[option] == "1"]
            else:
                workload_deviations = workload_deviations[workload_deviations[option] == "0"]
                workload_performance = workload_performance[workload_performance[option] == "0"]

        return 2.0 * max(float(
            workload_deviations[workload_deviations["revision"] == split_releases[0]]["performance"].iloc[0]) * float(
            workload_performance[workload_performance["revision"] == split_releases[0]]["performance"].iloc[0]),
                         float(workload_deviations[workload_deviations["revision"] == split_releases[1]][
                                   "performance"].iloc[0]) * float(
                             workload_performance[workload_performance["revision"] == split_releases[1]][
                                 "performance"].iloc[0]))

    def find_affected_terms(self, revisions: str, configuration: str, workload: str) -> List[str]:
        all_terms = self.affected_configurations_per_term_and_release[revisions][workload]
        selected_options = configuration.split(" ")

        # Search for the most general one
        affected_terms: Dict[int, List[str]] = dict()

        for term in all_terms.keys():
            options = term.split(" * ")
            ignore_term = False
            for option in options:
                if option not in selected_options:
                    ignore_term = True
                    break
            if ignore_term:
                continue
            number_affected_configurations = int(
                self.affected_configurations_per_term_and_release[revisions][workload][term])
            if number_affected_configurations not in affected_terms:
                affected_terms[number_affected_configurations] = list()
            affected_terms[number_affected_configurations].append(term)

        result = []
        for number_affected_configurations in sorted(affected_terms.keys()):
            for affected_term in affected_terms[number_affected_configurations]:
                result.append(affected_term)
        return result
