import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import pandas as pd


class WorkloadFrequencyAnalyzer:

    def __init__(self):
        self.workload_frequency_per_change = None

    def perform_analysis(self, output_path: str, input_path: str) -> None:
        with open(os.path.join(input_path, 'changes_detected_by_workloads.json'), 'r') as changes_file:
            self.workload_frequency_per_change = json.load(changes_file)

        performance_change_frequency: Dict[str, int] = dict()
        for releases in sorted(self.workload_frequency_per_change.keys()):
            for configuration in self.workload_frequency_per_change[releases].keys():
                performance_change_frequency[f"{releases}; {configuration}"] = len(
                    self.workload_frequency_per_change[releases][configuration])

        # sorted_performance_change_frequency = sorted(performance_change_frequency.items(), key=lambda x: x[1])

        max_workloads = max(performance_change_frequency.values())
        x_axis = range(0, max_workloads + 1)
        y_axis = [0] * (max_workloads + 1)
        average = 0.0
        for number_workloads in performance_change_frequency.values():
            y_axis[number_workloads] += 1
            average += number_workloads / max_workloads
        average /= len(performance_change_frequency.values())
        print(f"Average probability of picking a workload that identifies performance changes: {average * 100}%")

        df = pd.DataFrame(columns=["#Workloads", "Frequency"])
        df["#Workloads"] = x_axis
        df["Frequency"] = y_axis

        sns.set_color_codes("muted")
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(x="#Workloads", y="Frequency", data=df, color='b')

        fig = ax.get_figure()
        ax.set_ylabel("Frequency", fontsize=35)
        ax.set_xlabel("#Workloads", fontsize=35)
        ax.set_xlim(0, max_workloads + 0.5)
        plt.xticks(range(0, max_workloads + 1, 10))

        ax2 = ax.twiny()
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Workloads [%]", fontsize=35, labelpad=15)
        ax2.plot([],[])

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, 'workloadFrequency.pdf'))
        plt.close(fig)

