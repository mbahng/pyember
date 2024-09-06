from . import Model
from .. import Dataset

class Regression(Model): 

    def __init__(self): 
        self.state_dict = {}

    def fit(self, data: Dataset): 
        pass
