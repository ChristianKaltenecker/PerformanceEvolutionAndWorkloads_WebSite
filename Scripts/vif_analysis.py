#!/bin/env python3
import math
import sys
import os

import pandas as pd
from typing import List, Dict

from patsy.highlevel import dmatrices

from case_study import CaseStudy
from feature import Feature
from statsmodels.stats.outliers_influence import variance_inflation_factor


def print_usage():
    """Prints the usage of this script."""
    print("Usage: ./vif_analyzer <FeatureModelPath> <MeasurementFilePath> <ModelPath>")
    print("FeatureModelPath\t The path to the feature model")
    print("MeasurementFilePath\t The path to the measurement file (csv)")
    print("ModelPath\t\t The path to the models.txt containing one term of the performance model per line")


def main():
    """The main method of the VIF analysis."""
    feature_model_path: str = sys.argv[1]
    measurement_file_path: str = sys.argv[2]
    model_path: str = sys.argv[3]
    case_study = CaseStudy()
    vif_analyzer = VIFAnalyzer(os.path.dirname(feature_model_path), feature_model_path, measurement_file_path,
                               model_path)
    new_model: List[List[str]] = vif_analyzer.apply_multicollinearity_countermeasures()
    vif_analyzer.apply_iterative_vif(new_model, 'performance')


class VIFAnalyzer:
    """
    This script uses the feature model, measurements, and the model to
    (1) remove the terms from alternative features or mandatory features to reduce the terms that cause
    multicollinearity in performance-influence models,
    (2) an iterative VIF analysis is applied to detect further terms in the performance model that could cause
    multicollinearity, remove them from the model and print them out for further investigation.
    Finally, we obtain a performance-influence model containing only terms that cause no multicollinearity according
    to the VIF.
    """

    def __init__(self, case_study: CaseStudy, model_path: str) -> None:
        self.case_study: CaseStudy = case_study
        self.terms: List[List[str]] = self.read_model(model_path)

    @staticmethod
    def read_model(model_path: str) -> List[List[str]]:
        """
        This method reads in the model, where each line consists of one term (without coefficient).
        :param model_path: the path to the model
        :return: the list of terms, where each element consists of one or more terms.
        """
        if not os.path.exists(model_path):
            print("Model path not valid")
            exit(-1)
        term_list = []
        with open(model_path, 'r') as model_file:
            for line in model_file:
                line = line.replace('\n', '')
                features = line.split('*')
                features = [str.replace(feature, ' ', '') for feature in features]
                term_list.append(features)
        return term_list

    def apply_multicollinearity_countermeasures(self) -> List[List[str]]:
        """
        This method applies some countermeasures on the model to reduce multi-collinearity.
        That is, terms that are known to be perfectly multi-collinear are removed.
        :return: the list of terms without the ones that are known to cause multi-collinearity
        """
        # First, verify whether strictly mandatory features (i.e., no optional feature as direct/indirect parent)
        # appear more than once
        mandatory_count = 0
        new_terms = self.terms.copy()
        for term in self.terms:
            for feature_name in term:
                if self.case_study.features[feature_name].strictly_mandatory and \
                        self.case_study.features[feature_name].binary:
                    if mandatory_count == 0:
                        mandatory_count += 1
                    else:
                        print(f"Removing {term} since it contains a mandatory feature.")
                        new_terms.remove(term)
                        if len(term) > 1:
                            print(
                                f"Since there are multiple options in the term, the term '{feature_name}' is removed.")
                            term.remove(feature_name)
                            if term not in new_terms:
                                new_terms.append(term)
        # Afterward, remove the first alternative child from the model if it is contained
        term_level: Dict[int, List[List[str]]] = dict()
        for term in new_terms:
            level = len(term)
            if level not in term_level:
                term_level[level] = []
            term_level[level].append(term)
        for level in term_level.keys():
            terms = term_level[level]
            for term in terms:
                other_terms = new_terms.copy()
                other_terms.remove(term)
                for feature_name in term:
                    feature: Feature = self.case_study.features[feature_name]
                    if len(feature.alternatives) > 0:
                        # In this case, one of the features of the term is an alternative feature
                        # Now, check whether all other alternatives appear in the other terms of the same level
                        other_features_in_term = term.copy()
                        other_features_in_term.remove(feature_name)
                        other_features_in_alternative = feature.alternatives.copy()
                        for other_term in other_terms:
                            if all(element in other_term for element in other_features_in_term):
                                remaining_feature = [element for element in other_term if element not in
                                                     other_features_in_term]
                                if len(remaining_feature) > 0 and remaining_feature[0] in other_features_in_alternative:
                                    other_features_in_alternative.remove(remaining_feature[0])
                        if len(other_features_in_alternative) == 0:
                            print(f"Removing term {term} since all other alternative features are included.")
                            new_terms.remove(term)
                            terms.remove(term)
        return new_terms

    def apply_iterative_vif(self, model_to_check: List[List[str]], nfp: str, log_path: str = None, revision: str = None,
                            workload: str = None) -> List[List[str]]:
        """
        Applies an iterative VIF analysis. In each iteration, an additional term is included to the VIF analysis.
        Whenever the threshold of 5 is exceeded, the new term is removed. The removed term and the terms it is
        conflicting with is printed on the console.
        :param model_to_check: the model to check. It contains in each line a term of the performance-influence model
        :param nfp: the nfp to investigate
        :param log_path: the path to the log file where the conflicts are written
        :return: A reduced model where all conflicting model are already removed.
        """
        if len(model_to_check) < 2:
            print("The length of the given model is too short (less than 2)")
            exit(-1)
        log_file = None
        if log_path is not None:
            log_file = open(log_path, 'w')

        current_model = [model_to_check[0]]
        current_model_string = ['_'.join(model_to_check[0])]

        if revision is None:
            data = self.case_study.configurations
        else:
            data = self.case_study.configurations[self.case_study.configurations['revision'] == revision]

        if workload is not None:
            data = data[data['workload'] == workload]

        dataframe_for_vif = pd.DataFrame(data=data, columns=[model_to_check[0][0], nfp])
        for i in range(1, len(model_to_check)):
            current_model.append(model_to_check[i])
            current_model_string.append('__'.join(model_to_check[i]))
            # Append to the dataframe the needed information
            # The term could consist of one or more terms; therefore, we have to split it
            dataframe_for_vif[current_model_string[-1]] = self.case_study.configurations[model_to_check[i][0]]
            for j in range(1, len(model_to_check[i])):
                dataframe_for_vif[current_model_string[-1]] = dataframe_for_vif[current_model_string[-1]].astype(int) \
                                                              * self.case_study.configurations[model_to_check[i][j]] \
                                                                  .astype(int)
            # Construct the matrix for the VIF analysis
            y, X = dmatrices('performance ~ ' + '+'.join(current_model_string), data=dataframe_for_vif,
                             return_type='dataframe')

            # Exclude redundant columns -- no need for an VIF analysis
            if len(X.columns) < len(current_model_string):
                current_model = current_model[:-1]
                current_model_string = current_model_string[:-1]
                continue

            # Run the analysis
            vif = pd.DataFrame()
            vif['variable'] = X.columns
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            # Check if all VIFs are not infinity
            # If not, check with which terms the given term interferes.
            conflicting_terms = []
            vif_value = 0
            for index, row in vif.iterrows():
                # Values of nan or infinity mean that the variance is perfect
                if (math.isinf(row['VIF']) or math.isnan(row['VIF'])) and row['variable'] != "Intercept":
                    conflicting_terms.append(row['variable'])
                    vif_value = row['VIF']
            if len(conflicting_terms) > 0:
                print(f"Removing term {current_model_string[-1]} since it is conflicting with {str(conflicting_terms)}")
                if log_file is not None:
                    log_file.write(f"{current_model_string[-1]} ({vif_value}): {str(conflicting_terms)}\n")
                current_model = current_model[:-1]
                current_model_string = current_model_string[:-1]
        if log_file is not None:
            log_file.close()
        return current_model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print_usage()
        exit(-1)
    main()
