import torch

from .base import BaseModule


class Prenet(BaseModule):
    """
    Prenet module from Tacotron 2.
    """
    def __init__(self, config):
        super(Prenet, self).__init__()
        out_sizes = config['decoder_prenet_sizes']
        in_sizes = [config['decoder_rnn_dim']] + out_sizes[:-1]
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(in_size, out_size, bias=False)
            for (in_size, out_size) in zip(in_sizes, out_sizes)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = torch.nn.functional.dropout(linear(x).relu(), p=0.5, training=True)
        return x
