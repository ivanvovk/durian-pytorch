import torch

from .base import BaseModule
from .prenet import Prenet
from .utils import get_mask_from_lengths


class BaselineDecoder(BaseModule):
    """
    Much simplified decoder than the original one with Prenet.
    """
    def __init__(self, config):
        super(BaselineDecoder, self).__init__()
        self.rnn = torch.nn.GRU(
            input_size=2*config['encoder_rnn_dim'] + 1,
            hidden_size=config['decoder_rnn_dim'],
            num_layers=config['decoder_rnn_num_layers'],
            batch_first=True,
            bidirectional=False
        )
        self.rnn.flatten_parameters()
        self.conv1d = torch.nn.Conv1d(
            in_channels=config['decoder_rnn_dim'],
            out_channels=config['n_mel_channels'],
            kernel_size=1
        )
        
    def forward(self, aligned_features, output_lengths):
        """
        Decodes aligned acoustic features for further mel prediction.
        """
        outputs = torch.nn.utils.rnn.pack_padded_sequence(
            input=aligned_features, lengths=output_lengths,
            batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(outputs)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=outputs, batch_first=True
        )
        outputs = outputs.transpose(2, 1)
        pre_mel = self.conv1d(outputs)
        output_mask = get_mask_from_lengths(
            output_lengths, expand_multiple=pre_mel.shape[1]
        ).transpose(2, 1)
        pre_mel = pre_mel * output_mask
        return pre_mel, outputs
    
    def inference(self, aligned_features):
        outputs, _ = self.rnn(aligned_features)
        outputs = outputs.transpose(2, 1)
        pre_mel = self.conv1d(outputs)
        return pre_mel, outputs


class Decoder(BaseModule):
    """
    Strong self-conditionning decoder with Prenet.
    """
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.prenet = Prenet(config)
        decoder_rnn_input_size = config['decoder_prenet_sizes'][-1] + 2*config['embedding_dim'] + 1
        self._rnn_hidden_size = config['decoder_rnn_dim']
        self.rnns = torch.nn.ModuleList([
            torch.nn.LSTMCell(
                input_size=decoder_rnn_input_size,
                hidden_size=self._rnn_hidden_size,
                bias=True
            ),
            torch.nn.LSTMCell(
                input_size=self._rnn_hidden_size,
                hidden_size=self._rnn_hidden_size,
                bias=True
            )
        ])
        self.hidden_states = None
        self.conv1d = torch.nn.Conv1d(
            in_channels=config['decoder_rnn_dim'],
            out_channels=config['n_mel_channels'],
            kernel_size=1
        )
        
    def init_hidden_states(self, B):
        device = next(self.parameters()).device
        self.hidden_states = {
            rnn_idx: {
                'hx': torch.zeros(B, self._rnn_hidden_size, dtype=torch.float32).to(device),
                'cx': torch.zeros(B, self._rnn_hidden_size, dtype=torch.float32).to(device)
            } for rnn_idx in range(len(self.rnns))
        }
        
    def get_go_frame(self, B):
        device = next(self.parameters()).device
        return torch.zeros(B, self._rnn_hidden_size, dtype=torch.float32).to(device)
        
    def decode(self, decoder_input):
        outputs = decoder_input
        
        self.hidden_states[0]['hx'], self.hidden_states[0]['cx'] \
            = self.rnns[0](outputs, (self.hidden_states[0]['hx'], self.hidden_states[0]['cx']))
        
        self.hidden_states[1]['hx'], self.hidden_states[1]['cx'] \
            = self.rnns[1](self.hidden_states[0]['hx'], (self.hidden_states[1]['hx'], self.hidden_states[1]['cx']))
        return self.hidden_states[1]['hx']
    
    def forward(self, aligned_features, output_lengths):
        B, T = aligned_features.shape[:2]
        self.init_hidden_states(B)
        decoder_input = self.get_go_frame(B)
        
        outputs = []
        for frame_idx in range(T):
            decoder_input = self.prenet(decoder_input)
            decoder_input = torch.cat([decoder_input, aligned_features[:, frame_idx]], dim=1)
            outputs.append(self.decode(decoder_input))
            decoder_input = outputs[-1]
        outputs = torch.stack(outputs).transpose(0, 1).transpose(2,  1)
        outputs = self.conv1d(outputs)
        output_mask = get_mask_from_lengths(output_lengths, expand_multiple=outputs.shape[1]).transpose(2, 1)
        outputs = outputs * output_mask
        return outputs
    
    def inference(self, aligned_features):
        B, T = aligned_features.shape[:2]
        self.init_hidden_states(B)
        decoder_input = self.get_go_frame(B)
        
        outputs = []
        for frame_idx in range(T):
            decoder_input = self.prenet(decoder_input)
            decoder_input = torch.cat([decoder_input, aligned_features[:, frame_idx]], dim=1)
            outputs.append(self.decode(decoder_input))
            decoder_input = outputs[-1]
        outputs = torch.stack(outputs).transpose(0, 1).transpose(2,  1)
        outputs = self.conv1d(outputs)
        return outputs
