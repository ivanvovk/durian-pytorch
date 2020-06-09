import torch

from .base import BaseModule


class AlignmentModule(BaseModule):
    """
    Special module in DurIAN, which duplicates encoder hidden states
    with the correspodence to the outputs of duration model.
    """
    def __init__(self):
        super(AlignmentModule, self).__init__()
        
    def forward(self, encoded_inputs, alignments):
        durations = alignments.sum(dim=2)
        alignments = alignments.transpose(2, 1)
        outputs = alignments.bmm(encoded_inputs)
        B, max_target_len = encoded_inputs.shape[0], alignments.shape[1]
        position_encodings = torch.zeros(B, max_target_len, dtype=torch.float32).to(encoded_inputs)
        for obj_idx in range(B):
            positions = torch.cat([torch.linspace(0, 1, steps=int(dur), dtype=torch.float32)
                                   for dur in durations[obj_idx]]).to(encoded_inputs)
            position_encodings[obj_idx, :positions.shape[0]] = positions
        outputs = torch.cat([outputs, position_encodings.unsqueeze(dim=2)], dim=2)
        return outputs
    
    def inference(self, encoded_inputs, alignments):
        return self.forward(encoded_inputs=encoded_inputs, alignments=alignments)
