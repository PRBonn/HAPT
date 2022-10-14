import click
from os.path import join, dirname, abspath
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import datasets.datasets as datasets
from models.HAPT import Hapt as HAPT



@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)

def main(config,weights,checkpoint):
    cfg = yaml.safe_load(open(config))
    torch.manual_seed(cfg['experiment']['seed'])

    # Load data and model
    data = datasets.StatDataModule(cfg)
    
    if weights is None:
        model = HAPT(cfg)
    else: # this works only if we pre-traing the whole network (encoder + decoder)
        model = HAPT.load_from_checkpoint(weights,hparams=cfg)
       
    # Add callbacks:
    #lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver_pqp = ModelCheckpoint(dirpath='.',
                                 monitor='pqp',
                                 filename=cfg['experiment']['id']+'_best_pqp',
                                 mode='max',
                                 verbose=True,
                                 save_last=True)

    checkpoint_saver_pql = ModelCheckpoint(dirpath='.',
                                 monitor='pql',
                                 filename=cfg['experiment']['id']+'_best_pql',
                                 mode='max',
                                 verbose=True,
                                 save_last=False)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      max_epochs= cfg['train']['max_epoch'],
                      callbacks=[checkpoint_saver_pqp, checkpoint_saver_pql])
    # Train
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
