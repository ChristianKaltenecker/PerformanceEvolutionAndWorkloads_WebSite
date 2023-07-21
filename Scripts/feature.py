from typing import List
from typing import Dict


class Feature:
    name = ""
    binary = False
    mandatory = None
    strictly_mandatory = None
    parent = None
    children = []
    exclusions = []
    implications = []
    alternatives = []

    def __init__(self, name: str, parent, exclusions: List, implications: List, mandatory: bool = None) -> None:
        self.binary = mandatory is not None
        self.name = name
        self.mandatory = mandatory
        self.parent = parent
        self.implications = implications
        self.exclusions = exclusions

    def __str__(self):
        return self.name