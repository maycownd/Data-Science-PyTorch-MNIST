import unittest
from test_functions import *
import warnings


class Test_Model(unittest.TestCase):

    def test_deployment(self):
        self.assertTrue(checkdeployment())
    def test_training(self):
        self.assertTrue(checkTrainingMethod())
    def test_model_saving(self):
        self.assertTrue(checkModelSaving())
    def test_training_data_format(self):
        self.assertTrue(checkTrainingDataFormat())
    def test_accuracy(self):
        self.assertTrue(checkAccuracy())
        


# the following is not required if call by pytest instead of python
if __name__ == '__main__':
    unittest.main()
