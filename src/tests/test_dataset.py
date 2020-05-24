import unittest
from src.core.utils.dataset import Dataset

class Datasettests(unittest.TestCase):

    def setUp(self):
        self.image_path = "data/test/images/"
        self.annotations_path = "data/test/annotations/"

    def tearDown(self):
        pass

    def test_dataset_initialization(self):
        """
        Test that when the model is initialized without model name, the default model gets selected.
        """
        dataset = Dataset(image_path=self.image_path,
                          annotation_path=self.annotations_path,
                          transform=None)
        print("hi")
