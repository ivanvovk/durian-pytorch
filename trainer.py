import numpy as np

import torch

from utils import get_lr, show_message


class ModelTrainer(object):
    def __init__(
        self,
        config=None,
        optimizers={'backbone_model_opt': None, 'duration_model_opt': None},
        logger=None,
        criterion=None
    ):
        self._config = config
        self.optimizers = optimizers
        self.logger = logger
        self.criterion = criterion
        
        self.logger.save_model_config(self._config)
        
    def compute_loss(self, model, batch, training=True):
        model.train() if training else model.eval()
        outputs = model.forward(batch)
        losses = self.criterion(outputs, batch)
        loss_stats = self.criterion.loss_stats
        if training:
            return losses, loss_stats
        return losses, loss_stats, outputs
    
    def run_backward(self, model, losses):
        for loss in losses:
            loss.backward(retain_graph=True)
        grad_norm = self.gradient_apply_(model)
        return grad_norm
    
    def gradient_apply_(self, model):
        grad_norm = {
            'grad_norm/backbone_model_grad_norm': torch.nn.utils.clip_grad_norm_(
                parameters=model.backbone_model.parameters(),
                max_norm=self._config['grad_clip_threshold']
            ),
            'grad_norm/duration_model_grad_norm': torch.nn.utils.clip_grad_norm_(
                parameters=model.duration_model.parameters(),
                max_norm=self._config['grad_clip_threshold']
            )
        }
        for key in self.optimizers.keys():
            self.optimizers[key].step()
        model.zero_grad()
        return grad_norm
    
    def _should_validate(self, iteration):
        return (iteration % self._config['validation_step']) == 0
    
    def validate(self, iteration, model, dataloader, verbose=True):
        if self._should_validate(iteration):
            _val_stats = []
            for batch in dataloader:
                batch = model.parse_batch(batch)
                _, _loss_stats, outputs = self.compute_loss(model, batch, training=False)
                _val_stats.append(_loss_stats)
            stats = {}
            for key in _loss_stats.keys():
                stats[key] = torch.stack([batch_stats[key] for batch_stats in _val_stats]).mean()
            _B = outputs['postnet_outputs'].shape[0]
            random_idx_from_last_batch = np.random.choice(range(_B))
            pre_mel = outputs['pre_outputs'][random_idx_from_last_batch].detach().cpu()
            postnet_mel = outputs['postnet_outputs'][random_idx_from_last_batch].detach().cpu()
            mel_target = batch['mels_padded'][random_idx_from_last_batch].detach().cpu()
            alignment = outputs['alignments'][random_idx_from_last_batch].detach().cpu()
            alignment_target = batch['alignments_padded'][random_idx_from_last_batch].detach().cpu()
            stats.update({
                'image/pre_output': pre_mel,
                'image/postnet_output': postnet_mel,
                'image/mel_target': mel_target,
                'image/alignment_output': alignment,
                'image/alignment_target': alignment_target
            })
            self.log_validating(iteration, stats, verbose)

    def log_training(self, iteration, loss_stats, verbose=True):
        show_message(
            f"""Iteration {iteration} | Backbone loss {loss_stats['backbone_model/total_loss']} | Duration model {loss_stats['duration_model/total_loss']}""",
            verbose=verbose
        )
        self.logger.log(iteration, loss_stats={f'training/{key}': value
                                               for key, value in loss_stats.items()})
    
    def log_validating(self, iteration, loss_stats, verbose=True):
        show_message(
            f"""EVAL: Iteration {iteration} | Backbone loss {loss_stats['backbone_model/total_loss']} | Duration model {loss_stats['duration_model/total_loss']}""",
            verbose=verbose
        )
        self.logger.log(iteration, loss_stats={f'validating/{key}': value
                                               for key, value in loss_stats.items()})
        
    def get_current_lrs(self):
        lrs = {f'learning_rate/{key}': get_lr(self.optimizers[key])
               for key in self.optimizers.keys()}
        return lrs
        
    def _should_save_checkpoint(self, iteration):
        return (iteration % self._config['checkpoint_save_step']) == 0
        
    def save_checkpoint(self, iteration, model):
        if self._should_save_checkpoint(iteration):
            self.logger.save_checkpoint(iteration, model)
            
    def _finetune_submodule(self, model, submodule_name, checkpoint_filename, ignore=[]):
        d = torch.load(checkpoint_filename)
        model.load_state_dict({
            key: value for key, value in d.items() if submodule_name in key
                and key.split('.')[1] not in ignore
        }, strict=False)
    
    def finetune_backbone_model(self, model, checkpoint_filename, ignore=[]):
        model.finetune_backbone_model(checkpoint_filename, ignore)
        
    def finetune_duration_model(self, model, checkpoint_filename, ignore=[]):
        model.finetune_duration_model(checkpoint_filename, ignore)
