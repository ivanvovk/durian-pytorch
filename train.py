import json
import argparse

import torch

from data import Dataset, BatchCollate
from model import BaselineDurIAN, DurIAN
from loss import DurIANLoss
from logger import Logger
from trainer import ModelTrainer
from utils import show_message


def run(TTS_FRONTEND, TTS_CONFIG, args):
    show_message('Initializing data loaders...', verbose=args.verbose)
    batch_collate = BatchCollate(TTS_CONFIG)
    train_dataset = Dataset(TTS_CONFIG, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=TTS_CONFIG['batch_size'], shuffle=True,
        collate_fn=batch_collate, drop_last=True
    )
    val_dataset = Dataset(TTS_CONFIG, training=False)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=TTS_CONFIG['batch_size'], shuffle=True,
        collate_fn=batch_collate, drop_last=False
    )

    show_message('Initializing model...', verbose=args.verbose)
    TTS_CONFIG['n_symbols'] = len(batch_collate.text_frontend.SYMBOLS)
    model = TTS_FRONTEND(TTS_CONFIG)
    model.cuda()

    show_message('Initializing optimizers, loss and schedullers...', verbose=args.verbose)
    backbone_model_opt = torch.optim.Adam(
        params=model.backbone_model.parameters(),
        lr=TTS_CONFIG['learning_rate'],
        weight_decay=1e-6
    )
    backbone_model_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=backbone_model_opt, mode='min', factor=0.98, patience=500
    )
    duration_model_opt = torch.optim.Adam(
        params=model.duration_model.parameters(),
        lr=TTS_CONFIG['learning_rate']
    )
    duration_model_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=duration_model_opt, mode='min', factor=0.98, patience=1000
    )
    optimizers = {'backbone_model_opt': backbone_model_opt, 'duration_model_opt': duration_model_opt}
    criterion = DurIANLoss(TTS_CONFIG)

    show_message('Initializing logger and trainer...', verbose=args.verbose)
    logger = Logger(TTS_CONFIG['logdir'])
    trainer = ModelTrainer(config=TTS_CONFIG, optimizers=optimizers, logger=logger, criterion=criterion)

    try:
        iteration = 0
        for _ in range(TTS_CONFIG['n_epoch']):
            for batch in train_dataloader:
                # Training step
                batch = model.parse_batch(batch)
                losses, stats = trainer.compute_loss(model, batch, training=True)
                grad_norm = trainer.run_backward(model, losses=losses)
                stats.update(grad_norm)
                stats.update(trainer.get_current_lrs())
                trainer.log_training(iteration, stats)
                backbone_model_lr_scheduler.step(losses[0])
                duration_model_lr_scheduler.step(losses[1])
                
                # Evaluation step
                trainer.validate(iteration, model, val_dataloader)
                trainer.save_checkpoint(iteration, model)
                iteration += 1
    except KeyboardInterrupt:
        print('KeyboardInterrupt: training is stopped.')
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str)
    parser.add_argument('-b', '--baseline', required=False, default=False, type=bool)
    parser.add_argument('-v', '--verbose', required=False, default=True, type=bool)
    args = parser.parse_args()

    with open(args.config) as f:
        TTS_CONFIG = json.load(f)
    TTS_FRONTEND = BaselineDurIAN if args.baseline else DurIAN
    
    run(TTS_FRONTEND, TTS_CONFIG, args)
