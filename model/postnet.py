import torch

from .layers import Conv1dResidualBlock
from .base import BaseModule


class Postnet(BaseModule):
    def __init__(self, config, input_shape=None):
        super(Postnet, self).__init__()
        input_shape = config['decoder_rnn_dim'] if not input_shape else input_shape
        out_sizes = config['postnet_residual_block_sizes']
        in_sizes = [input_shape] + config['postnet_residual_block_sizes'][:-1]
        self.residual_blocks = torch.nn.Sequential(
            *[Conv1dResidualBlock(
                in_channels, out_channels,
                kernel_size=config['postnet_kernel_size'],
                dropout_p=config['postnet_dropout_p']
            ) for in_channels, out_channels in zip(in_sizes, out_sizes)]
        )
        self.conv1d = torch.nn.Conv1d(
            in_channels=out_sizes[-1],
            out_channels=config['n_mel_channels'],
            kernel_size=1
        )
        
    def forward(self, decoded_sequence):
        outputs = self.residual_blocks(decoded_sequence)
        outputs = self.conv1d(outputs)
        return outputs
    
    def inference(self, decoded_sequence):
        return self.forward(decoded_sequence=decoded_sequence)
