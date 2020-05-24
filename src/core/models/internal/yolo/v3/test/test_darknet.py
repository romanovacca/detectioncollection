import unittest
from src.core.models.internal.yolo.v3.darknet import *

class Darknettests(unittest.TestCase):

    def setUp(self):
        self.image_path = "data/test/images/"
        self.config_file_location = "../cfg/yolov3.cfg"


    def tearDown(self):
        pass

    # def test_parse_cfg(self):
    #     """
    #     Test that when the model is initialized without model name, the default model gets selected.
    #     """
    #     blocks = parse_cfg(self.config_file_location)
    #     self.assertIsInstance(blocks, list)
    #     self.assertIsInstance(blocks[0], dict)
    #     self.assertEqual(len(blocks), 108)

    def test_create_pytorch_modules(self):
        """
        Test that when the model is initialized without model name, the default model gets selected.
        """
        blocks = parse_cfg(self.config_file_location)
        modules = create_pytorch_modules(blocks)
        self.assertIsInstance(modules[0], dict)
        self.assertEqual(str(type(modules[1])), "<class 'torch.nn.modules.container.ModuleList'>")
        self.assertEqual(modules[0]["type"], "net")
