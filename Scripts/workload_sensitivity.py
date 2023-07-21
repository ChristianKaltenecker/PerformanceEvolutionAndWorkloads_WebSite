import process_workloads
from PerformanceEvolution.case_study import CaseStudy

import numpy as np
import os
import pandas as pd


class WorkloadSensitivityAnalyzer:
    def __init__(self):
        pass

    def process_data(self, case_study: CaseStudy, input_path: str, path: str) -> None:
        workloads = process_workloads.WORKLOADS[str(case_study.name)]
        releases = sorted(list(case_study.configurations["revision"].unique()))
        number_configs = int(len(case_study.configurations[case_study.configurations['revision'] == releases[-1]]) / len(workloads))
        heatmap_data = np.zeros((len(workloads), number_configs * (len(releases) - 1)))
        for workload in workloads:
            workload_pos = workloads.index(workload)
            changes_data = np.load(os.path.join(input_path, f"configuration_difference_{workload}"), allow_pickle=True)
            for release in range(len(changes_data)):
                for config_number in range(len(changes_data[release])):
                    if changes_data[release][config_number] < 0:
                        heatmap_data[workload_pos][release * len(changes_data[release]) + config_number] = -1
                    elif changes_data[release][config_number] > 0:
                        heatmap_data[workload_pos][release * len(changes_data[release]) + config_number] = 1
                    else:
                        heatmap_data[workload_pos][release * len(changes_data[release]) + config_number] = 0

        df = pd.DataFrame(heatmap_data)
        df.index = workloads
        df.index.names = ['workload']
        with open(os.path.join(path, "clustering.csv"), 'w') as heatmap_file:
            heatmap_file.write(df.to_csv())

        all_changes = 0
        found_number_changes = 0
        # For each term and release
        for i in df.columns:
            performance_changes = list(df[i].unique())
            # Remove 0 since this indicates no performance change
            if 0 in performance_changes:
                performance_changes.remove(0)
            all_changes += len(performance_changes)
            # Locate airport_p07airport2p2 and termesopt18strips_p18
            relevant_workloads = ['airport_p07airport2p2', 'satellite_p06pfile6', 'termesopt18strips_p18', 'mprime_prob02', 'logistics98_prob31', 'organicsynthesissplitopt18strips_p01', 'visitallopt11strips_problem06full', 'woodworkingopt08strips_p24', 'floortileopt11strips_optp01002', 'trucksstrips_p08']
            found_changes = list(df.loc[relevant_workloads][i].unique())
            if 0 in found_changes:
                found_changes.remove(0)
            found_number_changes += len(found_changes)

        print(f"Found changes by clusters: {found_number_changes} out of {all_changes} ({found_number_changes * 1.0 / all_changes * 100.0}%)")
