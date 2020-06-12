import torch

from .base import BaseModule, BaseDurIAN
from .encoder import CBHG
from .alignment import AlignmentModule
from .decoder import Decoder
from .postnet import Postnet
from .duration import DurationModel
from .utils import get_mask_from_lengths


class BackboneModel(BaseModule):
    """
    DurIAN backbone text-to-speech module implementation.
    """
    def __init__(self, config):
        super(BackboneModel, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=config['n_symbols'],
            embedding_dim=config['embedding_dim']
        )
        self.encoder = CBHG(config)
        self.alignment_module = AlignmentModule()
        self.decoder = Decoder(config)
        self.postnet = Postnet(config, input_shape=config['n_mel_channels'])
        
    def forward(self, inputs, alignments, input_lengths, output_lengths):
        """
        Performs forward pass of backbone TTS model with given alignment matrices,
        input and outputs lengths.
        """
        embedded_inputs = self.embedding(inputs)
        encoded_inputs = self.encoder(embedded_inputs, input_lengths)
        aligned_features = self.alignment_module(encoded_inputs, alignments)
        pre_outputs = self.decoder(aligned_features, output_lengths)
        postnet_outputs = pre_outputs + self.postnet(pre_outputs)
        output_mask = get_mask_from_lengths(
            output_lengths, expand_multiple=postnet_outputs.shape[1]
        ).transpose(2, 1)
        postnet_outputs = postnet_outputs * output_mask
        return pre_outputs, postnet_outputs
    
    def inference(self, inputs, alignments):
        """
        Performs forward pass of backbone TTS model with given alignment matrices.
        """
        embedded_inputs = self.embedding(inputs)
        encoded_inputs = self.encoder.inference(embedded_inputs)
        aligned_features = self.alignment_module.inference(encoded_inputs, alignments)
        pre_outputs = self.decoder.inference(aligned_features)
        postnet_outputs = pre_outputs + self.postnet.inference(pre_outputs)
        return pre_outputs, postnet_outputs


class DurIAN(BaseDurIAN):
    """
    Implementation of Duration Informed Attention Network (DurIAN).
    """
    def __init__(self, config):
        super(DurIAN, self).__init__()
        self.backbone_model = BackboneModel(config)
        self.duration_model = DurationModel(config)
    
    def forward(self, x):
        inputs, alignments, input_lengths, output_lengths \
            = x['sequences_padded'], x['alignments_padded'], x['input_lengths'], x['output_lengths']
        pre_outputs, postnet_outputs = self.backbone_model(
            inputs=inputs,
            alignments=alignments,
            input_lengths=input_lengths,
            output_lengths=output_lengths
        )
        alignments = self.duration_model(
            inputs=inputs,
            input_lengths=input_lengths,
            output_lengths=output_lengths
        )
        outputs = {
            'pre_outputs': pre_outputs,
            'postnet_outputs': postnet_outputs,
            'alignments': alignments
        }
        return outputs
    
    def inference(self, inputs, alignments=None):
        """
        Performs forward pass of whole TTS module with predicting phonemes durations if no ones provided.
        :param inputs: torch.LongTensor, sequence of phoneme embedding ids
        """
        if isinstance(alignments, type(None)):
            alignments, _ = self.duration_model.inference(inputs)
        pre_outputs, postnet_outputs = self.backbone_model.inference(
            inputs=inputs,
            alignments=alignments
        )
        outputs = {
            'pre_outputs': pre_outputs,
            'postnet_outputs': postnet_outputs,
            'alignments': alignments
        }
        return outputs
