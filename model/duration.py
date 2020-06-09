import torch

from .base import BaseModule
from .utils import get_mask_from_lengths


class DurationModel(BaseModule):
    def __init__(self, config):
        super(DurationModel, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=config['n_symbols'],
            embedding_dim=config['embedding_dim']
        )
        self.rnn = torch.nn.LSTM(
            input_size=config['embedding_dim'],
            hidden_size=config['duration_model_rnn_dim'],
            num_layers=config['duration_model_num_layers'],
            batch_first=True,
            bidirectional=True
        )
        self.rnn.flatten_parameters()
        self.linear = torch.nn.Linear(
            in_features=2*config['duration_model_rnn_dim'],
            out_features=config['max_decoder_steps']
        )
        
    def _forward_impl(self, inputs, input_lengths):
        outputs = self.embedding(inputs)
        outputs = torch.nn.utils.rnn.pack_padded_sequence(
            input=outputs, lengths=input_lengths, batch_first=True
        )
        outputs, _ = self.rnn(outputs)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=outputs, batch_first=True
        )
        outputs = self.linear(outputs)
        return outputs
    
    def _inference_impl(self, inputs):
        outputs = self.embedding(inputs)
        outputs, _ = self.rnn(outputs)
        outputs = self.linear(outputs)
        return outputs
    
    def _compute_weighted_forced_alignment(self, alignment):
        forced_alignment = torch.zeros_like(alignment)
        durations = torch.bincount(alignment.argmax(dim=0))
        start = 0
        for symbol_idx, dur in enumerate(durations):
            forced_alignment[symbol_idx, start:start+dur] \
                = torch.ones(dur, dtype=torch.float32).to(forced_alignment)
            start += dur
        return forced_alignment, durations
        
    def forward(self, inputs, input_lengths, output_lengths):
        """
        Calculates alignment between symbols and frames.
        :param inputs: sequence of phoneme embedding ids
        :param input_lengths: initial lengths of unpadded phoneme sequences
        :param input_lengths: initial lengths of unpadded mels
        :return: alignment map of shape (B, max_input_len, max_decoder_steps)
        """
        energies = self._forward_impl(inputs, input_lengths)
        energies = energies[:, :input_lengths[0], :output_lengths.max()]
        outputs = energies.sigmoid()
        
        input_mask = ~get_mask_from_lengths(input_lengths, expand_multiple=outputs.shape[2])
        output_mask = torch.ones_like(energies, dtype=torch.bool)
        output_mask_ = ~get_mask_from_lengths(output_lengths, expand_multiple=outputs.shape[1]).transpose(2, 1)
        output_mask[:, :, :output_mask_.shape[-1]] = output_mask_
        mask = ~(input_mask | output_mask)
        mask.to(outputs)
        outputs = outputs * mask
        return outputs
    
    def inference(self, inputs):
        """
        Calculates alignment and number of frames per symbol for inference purposes.
        :param inputs: sequence of phoneme embedding ids
        :return: alignment map and exact durations
        """
        energies = self._inference_impl(inputs)
        outputs = energies.sigmoid()
        eos_idx = list((outputs[0].sum(dim=0) > 0.1).cpu().numpy()).index(False)
        outputs = outputs[:, :, :eos_idx]
        outputs, durations = self._compute_weighted_forced_alignment(outputs[0])
        return outputs[None], durations
