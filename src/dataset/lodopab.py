"""
Provides LoDoPaBGroundTruthDataset.
"""
import numpy as np
from odl import uniform_discr
from .dataset import GroundTruthDataset
try:
    from dival.datasets.lodopab_dataset import LoDoPaBDataset
except ImportError:
    LoDoPaBDataset = None

class LoDoPaBGroundTruthDataset(GroundTruthDataset):
    """
    A wrapper dataset providing the ground truth images of LoDoPaB-CT.
    """
    def __init__(self, image_size=362, min_pt=(-0.13, -0.13), max_pt=(0.13, 0.13),
                 train_len=35820, validation_len=3522, test_len=3553):

        self.shape = (image_size, image_size)
        space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        self.train_len = train_len
        self.validation_len = validation_len
        self.test_len = test_len
        super().__init__(space=space)

    def generator(self, fold='train'):
        assert LoDoPaBDataset is not None, 'dival.datasets.lodopab.LoDoPaBDataset could not be imported, but is required by LoDoPaBDataset'
        dataset = LoDoPaBDataset(impl='astra_cpu')  # impl does not matter since we only use the images
        for i in range(self.get_len(fold=fold)):
            yield dataset.get_sample(i, part=fold, out=(False, True))[1]
