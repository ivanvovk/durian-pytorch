import torch


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
    
    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseDurIAN(BaseModule):
    def __init__(self):
        super(BaseDurIAN, self).__init__()
        self.backbone_model = None
        self.duration_model = None
        
    def parse_batch(self, batch):
        """
        Moves inputs to model's device.
        """
        device = next(self.parameters()).device
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch
        
    def _load_submodule_checkpoint(self, submodule_name, f, ignore=[]):
        d = torch.load(f)
        self.load_state_dict({
            key: value for key, value in d.items() if submodule_name in key
                and key.split('.')[1] not in ignore
        }, strict=False)
    
    def finetune_backbone_model(self, f, ignore=[]):
        self._load_submodule_checkpoint('backbone_model', f, ignore)
        
    def finetune_duration_model(self, f, ignore=[]):
        self._load_submodule_checkpoint('duration_model', f, ignore)
