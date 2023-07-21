import json
from typing import Tuple

import matplotlib.colors

from PerformanceEvolution.recall_analyzer import RecallAnalyzer
from analysis_levels import AnalysisLevels
import csv
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import pivot_table
import subprocess
from utilities import *
import getpass
from vif_analysis import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import process_workloads


class ConfigurationLevel(AnalysisLevels):
    name = "ConfigurationLevel"

    @staticmethod
    def execute_command(command: str) -> str:
        output = subprocess.getstatusoutput(command)
        status_code = output[0]
        message = output[1]

        # Throw an error if the command was not successfully executed
        if status_code != 0:
            raise RuntimeError(message)

        return message

    def __init__(self):
        self.error_sum = 0.0
        self.error_count = 0
        self.changed_configs: Dict[str, Dict[str, Dict[str, List[str]]]] = dict()
        self.configuration_level_changes: Dict[str, Dict[str, List[str]]] = dict()
        self.configuration_level_changes_with_direction: Dict[str, Dict[str, List[Tuple[str, str]]]] = dict()
        self.configuration_level_changes_per_config: Dict[str, Dict[str, List[str]]] = dict()
        self.configuration_level_changes_per_config: Dict[str, Dict[str, List[str]]] = dict()
        self.number_configuration_changes_per_release: Dict[str, int] = dict()

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

        with open(os.path.join(path, "config_changes.md"), 'w') as output_file:
            output_file.write("")

    def prepare(self, case_study: CaseStudy, input_path: str) -> None:
        pass

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
                model_file.write(term + "\n")

    def evaluate_metrics(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        pass

    def generate_plots(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        # If no directory 'models' is included, the performance-influence models for each release have to be
        #  created and learned
        input_path = os.path.join(input_path, case_study.name)
        workloads = process_workloads.WORKLOADS[str(case_study.name)]
        # Create one dataframe for each workload
        for workload in workloads:
            workload_configs = case_study.configurations[
                case_study.configurations[process_workloads.WORKLOAD_COLUMN_NAME] == workload]
            deviation_configs = case_study.deviations[
                case_study.configurations[process_workloads.WORKLOAD_COLUMN_NAME] == workload]
            workload_path = os.path.join(path, workload)
            if not os.path.exists(workload_path):
                os.mkdir(workload_path)
            self.generate_workload_plots(case_study, workload_configs, deviation_configs, workload_path, input_path,
                                         workload)
        self.generate_barplots_per_release(case_study, path)

    def generate_barplots_per_release(self, case_study: CaseStudy, output_path: str) -> None:
        configuration_changes = list()
        for releases in sorted(self.number_configuration_changes_per_release.keys()):
            first_release = releases.split(" - ")[0]
            number_total_configurations = len(
                case_study.configurations[case_study.configurations['revision'] == first_release])
            configuration_changes.append(
                float(self.number_configuration_changes_per_release[releases]) / number_total_configurations * 100.0)

        plt.rcParams['xtick.bottom'] = True
        plt.figure(figsize=(12, 11))
        sns.set_color_codes("muted")
        ax = sns.barplot(
            x=[label.replace("_", ".") for label in sorted(self.number_configuration_changes_per_release.keys())],
            y=configuration_changes,
            color='b')

        fig = ax.get_figure()
        ax.set_ylabel("Configurations [%]", fontsize=50)
        ax.set_xlabel("Releases", fontsize=50, labelpad=20)
        plt.xticks(rotation=45, ha='right')
        # ax.tick_params(labelsize=50)
        ax.set_ylim([0, 100])

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, 'configurationChanges.pdf'))
        plt.close(fig)

    def generate_workload_plots(self, case_study: CaseStudy, configurations: pd.DataFrame, deviations: pd.DataFrame,
                                path: str, input_path: str, workload: str):
        revisions = list(dict.fromkeys(configurations.revision))
        feature_names = case_study.get_all_feature_names()
        feature_names.append(process_workloads.WORKLOAD_COLUMN_NAME)
        mean_values = configurations.groupby(feature_names, sort=False).mean()
        mean_values.reset_index(inplace=True)

        mean_values = mean_values.sort_values('performance')
        mean_values.reset_index(inplace=True)
        number_configurations = len(mean_values)
        index_converter = dict(zip(mean_values['index'], mean_values.index))
        self.generate_difference_plots(configurations, deviations, case_study, index_converter, mean_values,
                                       number_configurations, path, list(reversed(revisions)), input_path, workload)

    def generate_difference_plots(self, all_configurations, all_deviations, case_study, index_converter, mean_values,
                                  number_configurations, path, revisions, input_path: str, workload: str):
        # (I) Plot them in a heatmap (x-axis = configurations; y-axis = revisions/releases; color = performance)
        # Data preparation
        # all_configurations.set_index(keys=feature_names, inplace=True)
        plot_data = np.zeros((len(revisions), int(len(mean_values))))
        plot_data[:] = np.nan
        deviation_values = np.zeros((len(revisions), int(len(mean_values))))
        deviation_values[:] = np.nan

        configurations = pd.DataFrame(mean_values)
        configurations = configurations.drop(columns=["index", "revision", "performance"], axis=1)
        for y in range(0, len(revisions)):
            # Filter for releases
            release_configurations = all_configurations[all_configurations.revision == revisions[y]]
            merged_configs = configurations.merge(release_configurations, on=configurations.columns.to_list(),
                                                  how='left', indicator=True)
            merged_configs = merged_configs.loc[merged_configs._merge == 'both', merged_configs.columns != '_merge']
            release_workload_configurations = all_deviations[all_deviations.revision == revisions[y]]
            merged_deviations = configurations.merge(release_workload_configurations,
                                                     on=configurations.columns.to_list(),
                                                     how='left', indicator=True)
            merged_deviations = merged_deviations.loc[
                merged_deviations._merge == 'both', merged_deviations.columns != '_merge']
            for idx, row in merged_configs.iterrows():
                plot_data[len(revisions) - 1 - y, idx] = float(row['performance'])
            for idx, row in merged_deviations.iterrows():
                deviation_values[len(revisions) - 1 - y, idx] = float(row['performance'])
        # Export plot_data
        deviation_values.dump(os.path.join(input_path, f"deviation_values_{workload}"))
        plot_data.dump(os.path.join(input_path, f"configuration_values_{workload}"))

        cmap = plt.get_cmap('Oranges')
        fontsize = 30
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        # cm = ax.pcolormesh(plot_data, cmap=cmap)
        cm = ax.pcolormesh(plot_data, cmap=cmap)  # , norm=matplotlib.colors.LogNorm())
        ax.set_title(case_study.name, fontsize=fontsize)
        ax.set_ylabel('Release', fontsize=fontsize)
        ax.set_xlabel('Configuration', fontsize=fontsize)
        ax.set_yticks(np.arange(0.5, len(revisions) + 0.5, step=1.0))
        ax.set_yticklabels(reversed(revisions), fontsize=fontsize)
        ax.get_xaxis().set_visible(False)
        cb = fig.colorbar(cm, ax=ax)
        cb.set_label('Performance [s]', fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
        fig = ax.get_figure()
        fig.tight_layout()
        super().create_directory(os.path.join(path, 'AbsolutePerformance'))
        fig.savefig(os.path.join(path, 'AbsolutePerformance', 'configurationsPerformance.pdf'))
        plt.close(fig)

        # Create a dataframe with the first performance value of each configuration in the first row
        # and the last performance value in the last row. Calculate the diff to find persisting regressions.
        # first_and_last_performance_values = np.zeros((2, int(len(mean_values))))
        # first_and_last_diff = np.zeros((int(len(mean_values))))
        # reverse_index_converter = {v: k for k, v in index_converter.items()}
        # for i in range(int(len(mean_values))):
        #     first_and_last_performance_values[0][i] = plot_data[np.isfinite(plot_data[:, i]), i][0]
        #     first_and_last_performance_values[1][i] = plot_data[-1, i]
        #     first_and_last_diff[i] = first_and_last_performance_values[1][i] - first_and_last_performance_values[0][i]
        #     if first_and_last_diff[i] / mean_values['performance'][i] * 100 > 10 and \
        #             first_and_last_diff[i] > 5:
        #         print(
        #             f"Configuration {self.pretty_print_config(mean_values.loc[i])[0]} from Workload {os.path.basename(path)} has a persisting performance regression.")

        # (II) Plot the differences between revisions (x-axis = configurations; y-axis = revisions/releases;
        # color = performance difference between revisions and alternatively performance difference between revision and
        # first revision)
        plot_data2 = np.copy(plot_data)
        for y in range(1, len(revisions)):
            for x in range(0, len(mean_values)):
                if plot_data[len(revisions) - 1 - y][x] == np.nan and \
                        plot_data[len(revisions) - y][x] != np.nan:
                    max_value = 2 * plot_data[len(revisions) - y][x] * \
                                deviation_values[len(revisions) - y][x]
                else:
                    max_value = 2 * max(plot_data[len(revisions) - 1 - y][x] * \
                                        deviation_values[len(revisions) - 1 - y][x],
                                        plot_data[len(revisions) - y][x] * \
                                        deviation_values[len(revisions) - y][x]
                                        )
                difference = - (plot_data[len(revisions) - 1 - y][x] - plot_data[len(revisions) - y][x])
                if abs(difference) < max_value or math.isnan(difference):
                    plot_data2[len(revisions) - 1 - y][x] = 0
                else:
                    plot_data2[len(revisions) - 1 - y][x] = difference
                    # Collect the data in a dictionary and write it in a markdown file later
                    self.add_change(f"{revisions[y]} - {revisions[y - 1]}", os.path.basename(path),
                                    mean_values.loc[x], difference / mean_values['performance'][x] * 100)

        plot_data2 = np.delete(plot_data2, len(revisions) - 1, axis=0)
        # Export plot_data2
        plot_data2.dump(os.path.join(input_path, f"configuration_difference_{workload}"))

        # Pick a colormap
        # cmap = plt.get_cmap('PRGn')
        cmap = plt.get_cmap('RdBu_r')
        cmap.set_bad(color='grey')
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        greatest_value = max(abs(np.nanmin(plot_data2)), abs(np.nanmax(plot_data2)))
        cm = ax.pcolormesh(plot_data2, cmap=cmap, vmin=-greatest_value, vmax=greatest_value)
        # cm = ax.pcolormesh(plot_data2, cmap=cmap)
        # ax.set_title(case_study.name, fontsize=fontsize)
        ax.set_ylabel('Release', fontsize=fontsize)
        ax.set_xlabel('Configuration', fontsize=fontsize)
        ax.set_yticks(range(0, len(revisions)))
        ax.set_yticklabels(reversed(revisions), fontsize=fontsize)
        ax.get_xaxis().set_visible(False)
        cb = fig.colorbar(cm, ax=ax, extend='both')
        cb.set_label('Performance [s]', fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
        fig = ax.get_figure()
        fig.tight_layout()
        self.create_directory(os.path.join(path, 'Difference'))
        fig.savefig(os.path.join(path, 'Difference', 'configurationsDifference.pdf'))
        plt.close(fig)

    def pretty_print_config(self, config: Dict[str, float], difference: float = None, machine_readable: bool = False) -> \
    Tuple[str, str]:
        if not machine_readable:
            result = "{"
            for k, v in config.items():
                if v == '1' and k != "root":
                    result = f"{result} {k},"
            result = f"{result}}}"
            diff = None
            if difference is not None:
                diff = "{:.2f}".format(difference)
                diff = f"{diff}%"
            return result, diff
        result = ""
        for k, v in config.items():
            if v == '1' and k != "root":
                result = f"{result} {k}"
        diff = None
        if difference is not None:
            diff = "{:.2f}".format(difference)
            diff = f"{diff}"
        return result.strip(), diff

    def add_change(self, release: str, workload: str, config: Dict[str, float], difference: float) -> None:
        pretty_printed_change = self.pretty_print_config(config, difference)
        machine_readable_change = self.pretty_print_config(config, difference, machine_readable=True)
        if release not in self.number_configuration_changes_per_release:
            self.number_configuration_changes_per_release[release] = 0
        self.number_configuration_changes_per_release[release] += 1
        if release not in self.changed_configs:
            self.changed_configs[release] = dict()
        if workload not in self.changed_configs[release]:
            self.changed_configs[release][workload] = dict()
        if pretty_printed_change[0] not in self.changed_configs[release][workload]:
            self.changed_configs[release][workload][pretty_printed_change[0]] = list()
        self.changed_configs[release][workload][pretty_printed_change[0]].append(pretty_printed_change[1])
        if release not in self.configuration_level_changes:
            self.configuration_level_changes[release] = dict()
            self.configuration_level_changes_per_config[release] = dict()
            self.configuration_level_changes_with_direction[release] = dict()
        if workload not in self.configuration_level_changes[release]:
            self.configuration_level_changes[release][workload] = list()
            self.configuration_level_changes_with_direction[release][workload] = list()
        if machine_readable_change[0] not in self.configuration_level_changes[release][workload]:
            self.configuration_level_changes[release][workload].append(machine_readable_change[0])
            self.configuration_level_changes_with_direction[release][workload].append(machine_readable_change)
        if machine_readable_change[0] not in self.configuration_level_changes_per_config[release]:
            self.configuration_level_changes_per_config[release][machine_readable_change[0]] = list()
        self.configuration_level_changes_per_config[release][machine_readable_change[0]].append(workload)

    def finish(self, path: str, input_path: str) -> None:
        if self.error_count > 0:
            print("Overall average error:" + "{:.2f}".format(self.error_sum / self.error_count))
        with open(os.path.join(path, 'config_changes.md'), 'a') as change_file:
            for release in sorted(self.changed_configs.keys()):
                change_file.write(f"\n# {release}\n\n")
                for workload in self.changed_configs[release].keys():
                    change_file.write(f"## {workload}\n")
                    for config in self.changed_configs[release][workload]:
                        for change in self.changed_configs[release][workload][config]:
                            change_file.write(f"* {config}{change}\n")
        with open(os.path.join(input_path, 'changed_configurations.json'), 'w') as changed_configurations:
            changed_configurations.write(json.dumps(self.configuration_level_changes))
        with open(os.path.join(input_path, 'changed_configurations_with_direction.json'),
                  'w') as changed_configurations:
            changed_configurations.write(json.dumps(self.configuration_level_changes_with_direction))
        with open(os.path.join(input_path, 'changes_detected_by_workloads.json'), 'w') as changes_file:
            changes_file.write(json.dumps(self.configuration_level_changes_per_config))
