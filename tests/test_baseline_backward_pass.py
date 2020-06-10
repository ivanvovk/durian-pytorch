import sys
sys.path.insert(0, '..')

import unittest
import json
import numpy as np

import torch

from model import BaselineDurIAN, DurIAN
from loss import DurIANLoss
from base import suite, BaseModelBackwardPassTest


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class BaselineDurIANBackwardPassMSETest(BaseModelBackwardPassTest):
    def __init__(self, *args, **kwargs):
        super(BaselineDurIANBackwardPassMSETest, self).__init__(*args, **kwargs)
        self.CLASS_TYPE = BaselineDurIAN
        with open('../configs/baseline.json') as f:
            self.config = json.load(f)
        self.config['n_symbols'] = 100
        self.config['solve_alignments_as_mse'] = True
        self.config['solve_alignments_as_bce'] = False
        self.criterion = DurIANLoss(self.config)


class BaselineDurIANBackwardPassBCETest(BaseModelBackwardPassTest):
    def __init__(self, *args, **kwargs):
        super(BaselineDurIANBackwardPassBCETest, self).__init__(*args, **kwargs)
        self.CLASS_TYPE = BaselineDurIAN
        with open('../configs/baseline.json') as f:
            self.config = json.load(f)
        self.config['n_symbols'] = 100
        self.config['solve_alignments_as_mse'] = False
        self.config['solve_alignments_as_bce'] = True
        self.criterion = DurIANLoss(self.config)


class BaselineDurIANBackwardPassJoinTest(BaseModelBackwardPassTest):
    def __init__(self, *args, **kwargs):
        super(BaselineDurIANBackwardPassJoinTest, self).__init__(*args, **kwargs)
        self.CLASS_TYPE = BaselineDurIAN
        with open('../configs/baseline.json') as f:
            self.config = json.load(f)
        self.config['n_symbols'] = 100
        self.config['solve_alignments_as_mse'] = True
        self.config['solve_alignments_as_bce'] = True
        self.criterion = DurIANLoss(self.config)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite([
        BaselineDurIANBackwardPassMSETest,
        BaselineDurIANBackwardPassBCETest,
        BaselineDurIANBackwardPassJoinTest
    ]))
