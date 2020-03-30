import unittest
from src.core.models.classsification import Model

class Classificationtests(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.model_known = Model(model="vgg16")

    def tearDown(self):
        del self.model
        del self.model_known

    def test_determine_model_none(self):
        """
        Test that when the model is initialized without model name, the default model gets selected.
        """
        self.assertEqual(self.model._model.__class__.__name__, 'ResNet')

    def test_determine_model_known_option(self):
        """
        Test that when the model is initialized with a known model name, the given model gets selected.
        """
        self.assertEqual(self.model_known._model.__class__.__name__, 'VGG')

    def test_determine_model_unknown_option(self):
        """
        Test that when the model is initialized with a unknown model name, a value error is raised.
        """
        with self.assertRaises(ValueError):
            model = Model("testing")

    def test_get_classes_config(self):
        pass