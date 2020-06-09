import sys
sys.path.insert(0, '..')

import unittest
import json
import numpy as np

import torch

from model import BaselineDurIAN, DurIAN


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class BaseModelInferenceTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BaseModelInferenceTest, self).__init__(*args, **kwargs)
        self.CLASS_TYPE = type(None)
        self.config = None

    def _initialize_baseline(self, backend='cpu'):
        model = self.CLASS_TYPE(self.config)
        model = model.to(backend)
        return model
    
    def _initialize_test_batch(self):
        batch_size = 4
        max_input_len, max_output_len = 20, 200
        
        sequences_padded = torch.LongTensor([
            np.random.choice(range(self.config['n_symbols']), size=max_input_len)
            for _ in range(batch_size)
        ])
        mels_padded = torch.FloatTensor(batch_size, max_output_len).normal_()
        alignments_padded = torch.zeros(
            batch_size, max_input_len, max_output_len, dtype=torch.float32
        )
        input_lengths = torch.LongTensor([20, 18, 16, 14])
        output_lengths = torch.LongTensor([200, 160, 180, 140])
        avg_dur = [
            int(output_length//input_length)
            for input_length, output_length in zip(input_lengths.numpy(), output_lengths.numpy())
        ]
        start = [0, 0, 0, 0]
        for obj_idx in range(batch_size):
            for symbol_idx in range(input_lengths[obj_idx]):
                start_ = start[obj_idx]
                avg_dur_ = avg_dur[obj_idx]
                alignments_padded[obj_idx, symbol_idx, start_:start_+avg_dur_] \
                    = torch.ones(avg_dur_, dtype=torch.float32)
                start[obj_idx] += avg_dur_
        
        batch = {
            'sequences_padded': sequences_padded,
            'mels_padded': mels_padded,
            'alignments_padded': alignments_padded,
            'input_lengths': input_lengths,
            'output_lengths': output_lengths
        }
        return batch
    
    def _perform_forward_pass(self, batch, model):
        batch = model.parse_batch(batch)
        outputs = model.forward(batch)
        return outputs

    def _check_model_outputs(self, outputs):
        self.assertIsInstance(outputs, dict)
        for key in outputs.keys():
            self.assertIsInstance(outputs[key], torch.Tensor)

    def test_model_initialization(self):
        self.assertIsInstance(self._initialize_baseline(), torch.nn.Module)

    def test_cpu_forward_pass(self):
        test_batch = self._initialize_test_batch()
        model = self._initialize_baseline(backend='cpu')
        outputs = self._perform_forward_pass(test_batch, model)
        self._check_model_outputs(outputs)
    
    def test_gpu_forward_pass(self):
        self.assertTrue(
            torch.cuda.is_available(),
            msg="No CUDA backend detected on your machine."
        )
        test_batch = self._initialize_test_batch()
        model = self._initialize_baseline(backend='cuda')
        outputs = self._perform_forward_pass(test_batch, model)
        self._check_model_outputs(outputs)


class BaselineDurIANInferenceTest(BaseModelInferenceTest):
    def __init__(self, *args, **kwargs):
        super(BaselineDurIANInferenceTest, self).__init__(*args, **kwargs)
        self.CLASS_TYPE = BaselineDurIAN
        with open('../configs/baseline.json') as f:
            self.config = json.load(f)
        self.config['n_symbols'] = 100


class DurIANInferenceTest(BaseModelInferenceTest):
    def __init__(self, *args, **kwargs):
        super(DurIANInferenceTest, self).__init__(*args, **kwargs)
        self.CLASS_TYPE = DurIAN
        with open('../configs/default.json') as f:
            self.config = json.load(f)
        self.config['n_symbols'] = 100


def suite():
    test_classes_to_run = [BaselineDurIANInferenceTest, DurIANInferenceTest]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    suite = unittest.TestSuite(suites_list)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
