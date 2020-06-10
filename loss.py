import torch

from model.utils import get_mask_from_lengths


class MaskedL2Loss(torch.nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()
        self._backbone_l2_loss = torch.nn.MSELoss(reduction='none')
        
    def forward(self, outputs, targets, mask):
        loss = self._backbone_l2_loss(outputs, targets)
        loss.masked_fill_(mask, value=0.0)
        return loss.mean()
    
    
class DurIANLoss(torch.nn.Module):
    def __init__(self, config):
        super(DurIANLoss, self).__init__()
        self.solve_alignments_as_bce = config['solve_alignments_as_bce']
        self.solve_alignments_as_mse = config['solve_alignments_as_mse']
        assert self.solve_alignments_as_bce or self.solve_alignments_as_mse
        self._joint_minimization = self.solve_alignments_as_bce and self.solve_alignments_as_mse
        self.loss_stats_ = None
    
    @property
    def loss_stats(self):
        return self.loss_stats_
        
    def forward(self, outputs, x):
        # Mel-spectrogram prediction model loss
        pre_outputs, postnet_outputs = outputs['pre_outputs'], outputs['postnet_outputs']
        mel_targets, output_lengths = x['mels_padded'], x['output_lengths']
        output_mask = ~get_mask_from_lengths(output_lengths, expand_multiple=mel_targets.shape[1]).transpose(2, 1)
        pre_loss = MaskedL2Loss()(pre_outputs, mel_targets, output_mask)
        postnet_loss = MaskedL2Loss()(postnet_outputs, mel_targets, output_mask)
        backbone_model_loss = pre_loss + postnet_loss
        
        # Duration model loss
        # Firstly, alignment BCE loss
        duration_model_alignments = outputs['alignments']
        alignments_targets = x['alignments_padded']
        alignment_loss = torch.nn.BCELoss()(
            duration_model_alignments, alignments_targets
        )
        # Sencondly, durations MSE loss
        durations_predicted = duration_model_alignments.sum(dim=2)
        durations_targets, input_lengths = alignments_targets.sum(dim=2), x['input_lengths']
        input_mask = ~get_mask_from_lengths(input_lengths, expand_multiple=1).squeeze(dim=2)
        durations_loss = MaskedL2Loss()(durations_predicted, durations_targets, input_mask)
        durations_loss_coef = 1e-5 if self._joint_minimization else 1
        duration_model_loss = \
            (durations_loss_coef * durations_loss if self.solve_alignments_as_mse else 0) \
            + (alignment_loss if self.solve_alignments_as_bce else 0)
        
        self.loss_stats_ = dict()
        self.loss_stats_['backbone_model/pre_loss'] = pre_loss
        self.loss_stats_['backbone_model/postnet_loss'] = postnet_loss
        self.loss_stats_['backbone_model/total_loss'] = backbone_model_loss
        self.loss_stats_['duration_model/durations_loss'] = durations_loss
        self.loss_stats_['duration_model/alignment_loss'] = alignment_loss
        self.loss_stats_['duration_model/total_loss'] = duration_model_loss
        return backbone_model_loss, duration_model_loss
