import os 
import math
import time
import argparse
import logging

import torch
from torch import nn

import lightning as pl 
from lightning.pytorch.utilities import rank_zero
pl.seed_everything(42)

from utils.optim import AdamWScale
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)

from transformers import AutoConfig, AutoTokenizer
from transformers import T5ForConditionalGeneration

from utils import params_count
from utils import seq2seq_datamodule




class Seq2seqDataModuleMixtral(seq2seq_datamodule.PretrainingDataModuleMixtral):
    pass

class Seq2seqDataModuleChatgpt(seq2seq_datamodule.PretrainingDataModuleChatgpt):
    pass
    pass

class Seq2seqDataModuleLlama(seq2seq_datamodule.PretrainingDataModuleLlama):
    pass

class LightningModule(pl.LightningModule):
    def __init__(self, 
                 learning_rate: float=2e-4,
                 weight_decay: float=0.,
                 warmup_steps: int=0,
                 final_cosine: float=1e-5,
                 output_dir: str='',
                 model_name_or_path: str='',
                ):

        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.warmup_steps  = warmup_steps
        self.final_cosine  = final_cosine
        self.output_dir    = output_dir
        self.model_name_or_path = model_name_or_path
        
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.tokenizer.add_tokens(['[NL]'])

        self.model  = T5ForConditionalGeneration.from_pretrained(self.model_name_or_path, config=self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        print('---------------------------------------------')
        print(self.model_name_or_path)
        print('total params_count:', params_count(self.model))
        print('---------------------------------------------')

        self.validation_step_outputs = []

    # TODO: for multi-device training
    # @rank_zero
    def save_model(self, loss=None, model_name=None):
        if model_name is None:
            model_name = f'step={self.global_step:05d},loss={loss:.4f}'

        dir_name = os.path.join(self.output_dir, 'model', model_name)
        print(f'## save model to {dir_name}')
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.config.pt_loss = loss
        self.model.save_pretrained(dir_name)
        self.tokenizer.save_pretrained(dir_name)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('valid_loss', loss, sync_dist=True)
        self.validation_step_outputs.append(loss)

    def configure_optimizers(self):
        # https://github.com/PiotrNawrot/nanoT5/blob/main/nanoT5/utils/model_utils.py#L222
        no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=self.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches - self.warmup_steps,
            eta_min=self.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[self.warmup_steps]
        )
        scheduler = {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    def on_validation_end(self):
        valid_loss = torch.stack(self.validation_step_outputs).mean().item()
        # TODO
        # rank_zero({'valid_loss': valid_loss})
        print({'valid_loss': valid_loss})
        self.save_model(loss=valid_loss)
        self.validation_step_outputs.clear()

    def on_train_end(self):
        self.save_model(model_name='final_model')



def cli_main():
    from lightning.pytorch.cli import LightningCLI
    
    # no effect
    # torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(LightningModule)


if __name__ == '__main__':
    cli_main()