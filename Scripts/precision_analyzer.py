import json
import os
from typing import Dict, List, Tuple

import numpy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from PerformanceEvolution.case_study import CaseStudy


class PrecisionAnalyzer:

    def __init__(self):
        # Dict[str (release), Dict[str (workload), List[Tuple[str (configurations), str (change)]]]]
        self.configurations = None
        # Dict[str (release), Dict[str (workload), List[Tuple[str (options), bool (speed up)]]]]
        self.options = None

    def perform_analysis(self, path: str, case_study: CaseStudy, models_path: str, input_path: str) -> None:
        with open(os.path.join(input_path, 'changed_configurations_with_direction.json'), 'r') as changed_configurations:
            self.configurations = json.load(changed_configurations)
        with open(os.path.join(input_path, 'changed_options_with_direction.json'), 'r') as changed_options:
            self.options = json.load(changed_options)

        # For each release, workload, and option-level change:
        # Determine the affected configurations on the configuration level
        # Determine whether one of these configurations are included in the configuration level changes

        total_changes = 0
        confirmed_direction = 0
        confirmed_changes = 0
        metric_change = 0
        conf_space_change = 0
        model_with_high_error = 0
        low_performance = 0
        workloads_confirmed = dict()
        workloads_total = dict()
        exclude_timeout_configurations = case_study.configurations[case_study.configurations["performance"] != 1800]
        for release in self.options.keys():
            split_release = release.split(" - ")
            for workload in self.options[release]:
                # Calculate the mean values per workload
                workload_configurations = exclude_timeout_configurations[
                    exclude_timeout_configurations["workload"] == workload]
                mean_performance = (numpy.mean(
                    workload_configurations[workload_configurations["revision"] == split_release[0]][
                        "performance"]),
                                    numpy.mean(workload_configurations[
                                                   workload_configurations["revision"] == split_release[1]][
                                                   "performance"]))
                workload_deviations = case_study.deviations[case_study.deviations["workload"] == workload]
                mean_deviation = (numpy.mean(
                    workload_deviations[workload_deviations["revision"] == split_release[0]]["performance"]),
                                  numpy.mean(workload_deviations[
                                                 workload_deviations["revision"] == split_release[1]][
                                                 "performance"]))

                total_changes += len(self.options[release][workload])
                if workload not in workloads_confirmed:
                    workloads_confirmed[workload] = 0
                    workloads_total[workload] = 0
                workloads_total[workload] += len(self.options[release][workload])
                for changed_term, speed_up in self.options[release][workload]:
                    if self.is_affected_term_in_configuration_level(release, workload, changed_term, speed_up=speed_up == "True"):
                        confirmed_direction += 1
                    if self.is_affected_term_in_configuration_level(release, workload, changed_term):
                        confirmed_changes += 1
                        workloads_confirmed[workload] += 1
                    else:
                        metric_or_space = self.has_change_with_another_metric_or_different_configuration_space(
                            case_study, release, workload, changed_term,
                            2 * max(mean_performance[0] * mean_deviation[0], mean_performance[1] * mean_deviation[1]),
                            speed_up=speed_up == "True")
                        if metric_or_space[0]:
                            metric_change += 1
                            # if metric_or_space[2]:
                            #     confirmed_direction += 1
                        elif metric_or_space[1]:
                            conf_space_change += 1
                        elif workload == "visitallopt11strips_problem06full" or workload == "miconic_s132":
                            model_with_high_error += 1
                        elif numpy.mean(mean_performance) < 0.1:
                            low_performance += 1
        self.create_latex_table_for_recall_per_workload(workloads_confirmed, workloads_total, path)
        precision = (confirmed_changes * 1.0) / (total_changes * 1.0) * 100.0
        affected_by_conf_space_change = (conf_space_change * 1.0) / (total_changes * 1.0) * 100.0
        affected_by_different_metric = (metric_change * 1.0) / (total_changes * 1.0) * 100.0
        affected_by_high_error = (model_with_high_error * 1.0) / (total_changes * 1.0) * 100.0
        indicating_low_performance = (low_performance * 1.0) / (total_changes * 1.0) * 100.0
        precision_direction = (confirmed_direction * 1.0) / (total_changes * 1.0) * 100.0
        print(f"Number option changes: {total_changes}")
        print(f"Precision: {precision}%")
        print(f"Affected by different metric: {affected_by_different_metric}%")
        print(f"Affected by configuration space change: {affected_by_conf_space_change}%")
        print(f"Affected by model with high error: {affected_by_high_error}%")
        print(f"Indicating low performance: {indicating_low_performance}%")
        print(f"Precision with direction: {precision_direction}%")

    def create_latex_table_for_recall_per_workload(self, confirmed_changes: Dict[str, int],
                                                   total_changes: Dict[str, int], path: str) -> None:
        workloads = list()
        recall_values = list()
        for workload in sorted(confirmed_changes.keys()):
            workloads.append(workload.replace("_", "\\_"))
            recall_values.append(float(confirmed_changes[workload]) / float(total_changes[workload]) * 100.0)
        df = pd.DataFrame(dict(Workload=workloads, Precision=recall_values))

        # Retrieve standard deviation for dissertation
        print(f"Standard deviation of precision: {np.std(df['Precision'])}%")

        fig = plt.figure(figsize=(5, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim([0, 100])
        sns.violinplot(data=df, y="Precision", linewidth=3, cut=0)
        ax.set_ylabel("Precision [%]")
        fig.tight_layout()
        fig.savefig(os.path.join(path, 'precision_violin.pdf'))
        plt.close(fig)
        # Convert the changes in the configuration level
        with open(os.path.join(path, 'precision_per_workload.tex'), 'w') as tex_file:
            tex_file.write(df.to_latex(index=False, float_format="${:.2f}\\%$".format))

    def has_optional_option(self, case_study: CaseStudy, term: str) -> bool:
        for option in term.split(" * "):
            if not case_study.features[option].mandatory:
                return True
        return False

    def has_change_with_another_metric_or_different_configuration_space(self, case_study: CaseStudy, releases: str,
                                                                        workload: str, term: str, threshold: float,
                                                                        speed_up: bool) -> Tuple[bool, bool, bool]:
        affected_configurations = case_study.configurations[case_study.configurations["workload"] == workload]
        for option in term.split(" * "):
            affected_configurations = affected_configurations[affected_configurations[option] == "1"]
        split_release = releases.split(" - ")
        first_release_configurations = affected_configurations[affected_configurations["revision"] == split_release[0]]
        first_release_configurations.reset_index(inplace=True)
        second_release_configurations = affected_configurations[affected_configurations["revision"] == split_release[1]]
        second_release_configurations.reset_index(inplace=True)
        found = False
        found_direction = False
        if len(first_release_configurations) == len(second_release_configurations):
            for index, row in first_release_configurations.iterrows():
                change = float(first_release_configurations.loc[index]["performance"]) - float(
                        second_release_configurations.loc[index]["performance"])
                if abs(change) > threshold:
                    found = True
                    if (speed_up and float(change) < 0) or (not speed_up and float(change) > 0):
                        found_direction = True
                        break

        return found, len(first_release_configurations) != len(second_release_configurations), found_direction

    def is_affected_term_in_configuration_level(self, release: str, workload: str,
                                                term: str, speed_up: bool = None) -> bool:
        # Filter the configurations from the configuration level corresponding to the term
        affected_configurations = []
        if release not in self.configurations or workload not in self.configurations[release]:
            return False
        for configuration, change in self.configurations[release][workload]:
            configuration_options = configuration.split(" ")
            # Each option of the term has to be in the current configuration
            not_found = False
            for option in term.split(" * "):
                if option not in configuration_options:
                    not_found = True
            if not_found:
                continue
            if speed_up is None:
                affected_configurations.append(configuration)
            elif (speed_up and float(change) < 0) or (not speed_up and float(change) > 0):
                affected_configurations.append(configuration)
        # If there is overlap, return true
        return len(affected_configurations) > 0
