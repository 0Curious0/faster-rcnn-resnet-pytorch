import copy
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision.transforms import functional as TF

# creating a backbone class as we will be using different backbones in training
class Backbone(nn.Module): 
    def __init__(self, pretrained=True):
        # calling the super class of Backbone class
        super().__init__()

        # loading the resnet50 model with pretrained weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_model = resnet50(weights=weights)

        # base_model.children() returns an iterator over immediate children modules, so we can convert it to a list
        # removing the conv5_x layer as in the paper and the last 2 avg_pool, fc layers
        self.backbone = nn.Sequential(*list(base_model.children())[:-3])


    def forward(self, x):
        # passing the input through the backbone model
        x = self.backbone(x)
        return x


# Here we could have used transforms.Compose(lambda img : resize_shorter_side(img), ToTensor()) without returning the target
# But as a practice, we will create custom transform classes to handle both the image and its corresponding annotation dictionary.

# Create a custom transform class to convert PIL images to PyTorch tensors and apply the resize transformation 
class ToTensorTransform:
    def __call__(self, img, target):
        img = TF.to_tensor(img)
        return img, target

# Create a custom transform class to resize the shorter side of the image using resize_shorter_side function and update the annotation dictionary accordingly
class ResizeTransform:
    def __init__(self, short_side=600, max_side=1000):
        self.short_side = short_side
        self.max_side = max_side

    def __call__(self, img, target):
        return self.resize_shorter_side(img, target)

    # resize fn to resize the shorter side of the image to a specified size while maintaining the aspect ratio, and also update the annotation dictionary with the new dimensions of the image.
    def resize_shorter_side(self, img, target):
        h, w = img.size
        annotation_dict = copy.deepcopy(target)

        scale = self.short_side / min(w, h)
        if scale * max(w, h) > self.max_side:
            scale = self.max_side / max(w, h)
            
        new_w = int(w * scale)
        new_h = int(h * scale)

        annotation_dict["size"]["new_width"] = new_w
        annotation_dict["size"]["new_height"] = new_h

        return TF.resize(img, (new_h, new_w)), annotation_dict

# Create a composed transform class to apply multiple transformations sequentially
class ComposedTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for transform in self.transforms:
            img, target = transform(img, target)
        return img, target


backbone_transform = ComposedTransform([
    ResizeTransform(short_side=600, max_side=1000),
    ToTensorTransform(),
])
