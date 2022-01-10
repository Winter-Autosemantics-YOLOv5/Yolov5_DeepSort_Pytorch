import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

import sys
# so that init does not execute in the package
sys.path.append('deep_sort/deep/reid')
from torchreid import models
from torchreid import utils

model_zoo = {
    'osnet_x0_25': 'https://drive.google.com/u/0/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj&export=download',
    'osnet_x0_5': '',
    'osnet_x0_75': '',
    'osnet_x1_0': '',
}

def try_download_model(model_type):
    import gdown
    file_path = None
    
    if model_type in model_zoo.keys():
        filename = gdown(model_zoo[model_type])

    return file_path

class Extractor(object):
    def __init__(self, model_type, model_path=None, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.input_width = 128
        self.input_height = 256

        self.model = models.build_model(name=model_type, num_classes=1, loss='triplet')

        if model_path:
            utils.load_pretrained_weights(self.model, model_path)

        self.model.to(self.device)
        self.model.eval()

        logger = logging.getLogger("root.tracker")
        logger.info("Selected model type: {}".format(model_type))
        self.size = (self.input_width, self.input_height)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("osnet_x1_0")
    feature = extr(img)
    print(feature.shape)
