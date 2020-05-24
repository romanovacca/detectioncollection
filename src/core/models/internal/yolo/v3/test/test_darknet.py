import unittest
from src.core.models.internal.yolo.v3.darknet import *
from src.core.models.internal.yolo.v3.util import Handler


class Darknettests(unittest.TestCase):

    def setUp(self):
        self.image_path = "data/test/images/"
        self.config_file_location = "../cfg/yolov3.cfg"

    def tearDown(self):
        pass

    def test_parse_cfg(self):
        """
        Test that when the model is initialized without model name, the default model gets selected.
        """
        blocks = parse_cfg(self.config_file_location)
        self.assertIsInstance(blocks, list)
        self.assertIsInstance(blocks[0], dict)
        self.assertEqual(len(blocks), 108)

    def test_create_pytorch_modules(self):
        """
        Test that when the model is initialized without model name, the default model gets selected.
        """
        blocks = parse_cfg(self.config_file_location)
        modules = create_pytorch_modules(blocks)
        self.assertIsInstance(modules[0], dict)
        self.assertEqual(str(type(modules[1])), "<class 'torch.nn.modules.container.ModuleList'>")
        self.assertEqual(modules[0]["type"], "net")

    def test_model_initialization(self):
        model = Darknet(self.config_file_location)
        self.assertEqual(str(type(model)), "<class 'src.core.models.internal.yolo.v3.darknet.Darknet'>")

    def test_model_initialization(self):
        handler = Handler()
        inp = handler.get_test_input()

        model = Darknet(self.config_file_location)
        pred = model(inp, torch.cuda.is_available())
        self.assertEqual(pred.shape[0], 1)
        self.assertEqual(pred.shape[1], 22743)
        self.assertEqual(pred.shape[2], 85)
