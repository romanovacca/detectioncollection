import unittest
from src.core.Models import Model

class Modeltests(unittest.TestCase):
    model = Model()

    #del model

    def setUp(self):
        pass

    def test_determine_model_none(self):
        """
        Test that when the model is initialized without model name, the default model gets selected.
        """
        self.assertEqual(Modeltests.model._model.__class__.__name__, 'ResNet')

    def test_determine_model_known_option(self):
        """
        Test that when the model is initialized with a known model name, the given model gets selected.
        """
        model = Model("vgg16")
        self.assertEqual(model._model.__class__.__name__, 'VGG')

    def test_determine_model_unknown_option(self):
        """
        Test that when the model is initialized with a unknown model name, a value error is raised.
        """
        with self.assertRaises(ValueError):
            model = Model("testing")