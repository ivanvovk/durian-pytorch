import torch

from .layers import (
    Conv1dResidualBlock,
    BatchNormConv1d,
    Highway
)
from .base import BaseModule


class BaselineEncoder(BaseModule):
    """
    Simplified and weaker than original CBHG text encoder
    consisting of residual convolutional blocks and bidirectional GRU in the end.
    """
    def __init__(self, config):
        super(BaselineEncoder, self).__init__()
        out_sizes = config['encoder_residual_block_sizes']
        in_sizes = [config['embedding_dim']] + config['encoder_residual_block_sizes'][:-1]
        self.residual_blocks = torch.nn.Sequential(*[
            Conv1dResidualBlock(
                in_channels, out_channels,
                kernel_size=config['encoder_kernel_size'],
                dropout_p=config['encoder_dropout_p']
            ) for in_channels, out_channels in zip(in_sizes, out_sizes)
        ])
        self.rnn = torch.nn.GRU(
            input_size=config['encoder_residual_block_sizes'][-1],
            hidden_size=config['encoder_rnn_dim'],
            num_layers=config['encoder_rnn_num_layers'],
            batch_first=True,
            bidirectional=True
        )
        self.rnn.flatten_parameters()
        
    def forward(self, embedded_inputs, input_lengths):
        """
        Encodes sequence of embeddings.
        :param embedded_inputs: sequence of phoneme embeddings of shape (B, T, embd_dim)
        :param input_lengths: initial lengths of unpadded phoneme sequences
        :return: encoded sequences of shape (B, T, encoder_rnn_dim)
        """
        outputs = embedded_inputs.transpose(2, 1)
        outputs = self.residual_blocks(outputs).transpose(2, 1)
        outputs = torch.nn.utils.rnn.pack_padded_sequence(
            input=outputs, lengths=input_lengths, batch_first=True
        )
        outputs, _ = self.rnn(outputs)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=outputs, batch_first=True
        )
        return outputs
    
    def inference(self, embedded_inputs):
        """
        Encodes sequence of embeddings.
        :param embedded_inputs: sequence of phoneme embeddings of shape (B, T, embd_dim)
        :return: encoded sequences of shape (B, T, encoder_rnn_dim)
        """
        outputs = embedded_inputs.transpose(2, 1)
        outputs = self.residual_blocks(outputs).transpose(2, 1)
        outputs, _ = self.rnn(outputs)
        return outputs


class CBHG(BaseModule):
    """
    Well-performing encoder module from Tacotron 1.
    """
    def __init__(self, config):
        super(CBHG, self).__init__()
        self._input_size = config['embedding_dim']
        self.conv1d_bank = torch.nn.ModuleList([
            BatchNormConv1d(
                self._input_size, self._input_size,
                kernel_size=kernel_size, stride=1, padding=kernel_size//2
            ) for kernel_size in range(1, config['cbhg_max_kernel_size'] + 1)
        ])
        self.max_pool1d = torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        out_sizes = config['cbhg_projection_sizes']
        in_sizes = [config['cbhg_max_kernel_size'] * self._input_size] + out_sizes[:-1]
        self.conv1d_projections = torch.nn.ModuleList([
            BatchNormConv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            for (in_channels, out_channels) in zip(in_sizes, out_sizes)
        ])
        self.pre_highway = torch.nn.Linear(out_sizes[-1], self._input_size, bias=False)
        self.highways = torch.nn.ModuleList([
            Highway(self._input_size, self._input_size)
            for _ in range(config['cbhg_num_highway_layers'])
        ])
        self.rnn = torch.nn.GRU(
            input_size=self._input_size,
            hidden_size=self._input_size,
            num_layers=config['cbhg_rnn_num_layers'],
            batch_first=True,
            bidirectional=True
        )
        self.rnn.flatten_parameters()

    def forward(self, embedded_inputs, input_lengths):
        outputs = embedded_inputs.transpose(2, 1)
        
        # Concat conv1d bank outputs
        outputs = torch.cat([
            conv1d(outputs)[:, :, :outputs.shape[-1]]
            for conv1d in self.conv1d_bank
        ], dim=1)
        outputs = self.max_pool1d(outputs)[:, :, :outputs.shape[-1]]
        
        for conv1d in self.conv1d_projections:
            outputs = conv1d(outputs)
        outputs = outputs.transpose(1, 2)
        
        if outputs.shape[-1] != self._input_size:
            outputs = self.pre_highway(outputs)

        # Residual connection
        outputs += embedded_inputs
        for highway in self.highways:
            outputs = highway(outputs)
        
        outputs = torch.nn.utils.rnn.pack_padded_sequence(
            input=outputs, lengths=input_lengths, batch_first=True
        )
        outputs, _ = self.rnn(outputs)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=outputs, batch_first=True
        )
        return outputs
    
    def inference(self, embedded_inputs):
        outputs = embedded_inputs.transpose(2, 1)
        
        outputs = torch.cat([
            conv1d(outputs)[:, :, :outputs.shape[-1]] for conv1d in self.conv1d_bank
        ], dim=1)
        outputs = self.max_pool1d(outputs)[:, :, :outputs.shape[-1]]
        
        for conv1d in self.conv1d_projections:
            outputs = conv1d(outputs)
        outputs = outputs.transpose(1, 2)
        
        if outputs.shape[-1] != self._input_size:
            outputs = self.pre_highway(outputs)

        outputs += embedded_inputs
        for highway in self.highways:
            outputs = highway(outputs)
            
        outputs, _ = self.rnn(outputs)
        return outputs
