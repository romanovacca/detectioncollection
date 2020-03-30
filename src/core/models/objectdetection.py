import os
import torchvision
import torchvision.models as models

from src.core.config import config


class Model():
    def __init__(self, model=None, classes=None, device=None):
        """ Initializes a model that is predefined or manually added.
        Most models are taked from Pytorch's torchvision. These model
        can be for object detection purposes.
        """

        self._determine_model(model)

        self._configure_classes(classes)

        self._configure_device(device)

    def _determine_model(self, model):
        available_models = {
            "yolov3": models.resnet18()
        }

        if model in available_models:
            self._model = available_models[model]
        elif model == None:
            self._model = available_models["yolov3"]
        else:
            raise ValueError("The model that you have chosen is not in the current library.")

    def _configure_classes(self,classes):
        if not isinstance(classes,list) and not classes == None:
            raise TypeError

        if classes:
            print("true")
        else:
            self._classes = ["__background__"] + config["object_detection"]["default_classes"]
            self.classes = {label: index for index, label in enumerate(self._classes)}

    def _configure_device(self,device ):
        self.device = device if device else config["device"]["default"]