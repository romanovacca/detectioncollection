import os
import torchvision
import torchvision.models as models

from src.core.config import config


class Model():
    def __init__(self, model=None, classes=None, device=None):
        """ Initializes a model that is predefined or manually added.
        Most models are taked from Pytorch's torchvision. These model
        can be for classification/object detection purposes.
        """

        self._determine_model(model)

        if classes:
            print("true")
        else:
            self._classes = config["classification"]["resnet_classes"]

        self.classes_mapping = {label: index for index, label in enumerate(self._classes)}

    def _determine_model(self, model):
        available_models = {
            "resnet18": models.resnet18(),
            "alexnet": models.alexnet(),
            "vgg16": models.vgg16()
        }

        if model in available_models:
            self._model = available_models[model]
        elif model == None:
            self._model = available_models["resnet18"]
        else:
            raise ValueError("The model that you have chosen is not in the current library.")
        #print(f"Model chosen: {self._model.__class__.__name__}")

