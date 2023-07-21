[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

# Performance Prediction in the Presence of Workload Variability

This repository is the supplementary Web site for the chapter "Performance Prediction in the Presence of Workload Variability" in the dissertation submitted by Christian Kaltenecker. In this repository, we list further information to the paper.
Note that we have included the feature diagram and detailed information with regards to the precision and recall to the appendix of the dissertation.

## Plots

The plots that are obtained by executing our scripts, are included in different directories in this repository.
We explain them next.
* `ConfigurationLevel`: This directory contains all plots related to the performance changes of FastDownward at the configuration level.
All performance changes at the configuration level are included in the file [config_changes.md](./ConfigurationLevel/config_changes.md).
The number of configuration changes per release are included in the file [configurationChanges.pdf](./ConfigurationLevel/FastDownward/configurationChanges.pdf).
We have also included detailed plots to the performance changes at the configuration level in the directory [FastDownward](./ConfigurationLevel/FastDownward/).

* `OptionLevel`: This directory contains all plots related to performance changes of FastDownward at the option level.
The file [identified_changes.md](./OptionLevel/identified_changes.md) contains a detailed list of all performance changes.
Detailed plots to the performance changes at the option level are in the directory [FastDownward](./OptionLevel/FastDownward/).

* `Precision`: This directory contains the [plot](./Precision/precision_violin.pdf) with the precision values (RQ1.1) over all workloads.

* `Recall`: This directory contains the [plot](./Recall/recall_violin.pdf) with the recall values (RQ1.2) over all workloads.

* `Frequency`: This directory contains the [plot](./Frequency/workloadFrequency.pdf) about the workload frequency in RQ2.

* `PersistingRegressions`: This directory contains the [plot](./PersistingRegressions/persistingRegressions.pdf) related to the insight on how many releases regressions persist.

* `Clustering`: This directory contains the clustering of the workloads.
We provide a [plot](./Clustering/clustering.pdf) for two clusters and one [plot](./Clustering/silhouette.pdf) using the average silhouette width.

## Data

The data obtained by performance measurements of the subject system `FastDownward` are included in the directory [Measurement_Data](./Measurement_Data/).
This directory contains the feature model, the results of the performance measurements (i.e., the measurements in the file `measurements.csv` and the deviations in the file `deviations.csv`), and the models learned on each workload and in release in `models/models.csv`.

We use the given data in the scripts from the directory [Scripts](./Scripts/), which we explain next.

## Scripts

The directory [Scripts](./Scripts/) contains the scripts for identifying the performance changes at the configuration level and at the option level.
Therein, we use the variance inflation factor (VIF) analysis.
Furthermore, the directory contains other scripts to assess all data and plots used in the dissertation.

To execute our scripts, first make sure to install the required packages:
```
pip install -r ./Scripts/requirements.txt
```

After installing the required packages in the required versions, you can execute the scripts by providing the path to the data directory `../Measurement_Data/` and an output directory `/tmp/Output/`:
```
./execute_performance_analysis.py ../Measurement_Data/ /tmp/Output/
```

After executing the python scripts, R scripts have to be executed to obtain the clustered dendrogram.
Please install `R` on your system and check that `Rscript` is also available.
Afterwards, install the required packages:
```
Rscript ./InstallPackages.R
```

Then execute the python script using the output directory specified before:
```
Rscript ./generate_dendrogram.R /tmp/Output/
```