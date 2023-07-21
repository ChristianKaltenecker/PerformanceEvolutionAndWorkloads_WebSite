from case_study import CaseStudy
from recall_analyzer import RecallAnalyzer
import os


class AnalysisLevels:

    @staticmethod
    def create_directory(path: str) -> None:
        """
        Creates the given directory if it does not exist already.
        :param path: the path to create
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def get_name(self):
        return self.name

    def initialize_for_metrics(self, path: str):
        pass

    def prepare(self, case_study: CaseStudy, input_path: str) -> None:
        pass

    def evaluate_metrics(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        pass

    def generate_plots(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        pass

    def finish(self, path: str, input_path: str) -> None:
        pass
