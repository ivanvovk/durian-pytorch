import sys
sys.path.insert(0, '..')

import unittest
import json
import numpy as np

import torch

from model import BaselineDurIAN
from base import suite, BaseModelForwardPassTest


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class BaselineDurIANForwardPassTest(BaseModelForwardPassTest):
    def __init__(self, *args, **kwargs):
        super(BaselineDurIANForwardPassTest, self).__init__(*args, **kwargs)
        self.CLASS_TYPE = BaselineDurIAN
        with open('../configs/baseline.json') as f:
            self.config = json.load(f)
        self.config['n_symbols'] = 100


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite([BaselineDurIANForwardPassTest]))
