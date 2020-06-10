import sys
sys.path.insert(0, '..')

import unittest
import json
import numpy as np

import torch

from model import DurIAN
from base import suite, BaseModelForwardPassTest


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class DurIANForwardPassTest(BaseModelForwardPassTest):
    def __init__(self, *args, **kwargs):
        super(DurIANForwardPassTest, self).__init__(*args, **kwargs)
        self.CLASS_TYPE = DurIAN
        with open('../configs/default.json') as f:
            self.config = json.load(f)
        self.config['n_symbols'] = 100


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite([DurIANForwardPassTest]))
