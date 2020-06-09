import torch

from .base import BaseModule


class Conv1dResidualBlock(BaseModule):
    """
    Special module for simplified version of Encoder class.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout_p=0.5):
        super(Conv1dResidualBlock, self).__init__()
        self.main_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.Dropout(p=dropout_p)
        )
        self.conv1d_residual = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        outputs = self.main_block(x)
        outputs += self.conv1d_residual(x)
        return outputs


class BatchNormConv1d(BaseModule):
    """
    Convolutional block with ReLU activation followed by batch normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BatchNormConv1d, self).__init__()
        self.convolution = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=padding, bias=False
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.convolution(x)


class Highway(BaseModule):
    """
    Implementation as described
    in https://arxiv.org/pdf/1505.00387.pdf.
    """
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = torch.nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = torch.nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)

    def forward(self, inputs):
        H = self.H(inputs).relu()
        T = self.T(inputs).sigmoid()
        outputs = H * T + inputs * (1 - T)
        return outputs
