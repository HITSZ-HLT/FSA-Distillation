import os 
import time
import argparse
import logging

import random
import torch
import lightning as pl 
pl.seed_everything(42)

from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer

from utils import params_count, load_json
from utils.atsa import Result
from sklearn.model_selection import train_test_split



class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str='',
        max_seq_length: int = -1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        data_dir: str = '',
        dataset: str = '',
        subname: str = '',
        training_data_prop: float = 1.,
        seed: int = 42,
        test_type: str = 'normal', # ('normal', 'implicit', 'mams')
    ):

        super().__init__()

        self.max_seq_length     = max_seq_length
        self.train_batch_size   = train_batch_size
        self.eval_batch_size    = eval_batch_size
        self.training_data_prop = training_data_prop
        self.dataset            = dataset
        self.seed               = seed
        self.test_type          = test_type

        self.model_name_or_path = model_name_or_path
        if self.model_name_or_path in ('', 'subname'):
            self.model_name_or_path = load_json('bash/model_name.json')[subname]

        print('使用的模型地址:', self.model_name_or_path)
        self.data_dir = data_dir if dataset == '' else os.path.join(data_dir, dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def load_dataset(self):



        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name   = os.path.join(self.data_dir, 'dev.json')
        test_file_name  = os.path.join(self.data_dir, 'test.json')

        train_examples = load_json(train_file_name)
        dev_examples   = load_json(dev_file_name)
        test_examples  = load_json(test_file_name)

        hard_file_name = os.path.join(self.data_dir, 'hard_test.json')
        if os.path.exists(hard_file_name):
            test_examples.extend(load_json(hard_file_name))

        print('-'*100)
        print('total train dataset length:')
        print(len(train_examples))

        if self.training_data_prop < 1:
            k = int(len(train_examples) * self.training_data_prop)
            train_examples = random.sample(train_examples, k=k)

        print('data prop is ', self.training_data_prop)
        print('used dataset length:')
        print(len(train_examples))


        def is_implicit(example):
            """
            Check if any aspect in the example has 'opinion_words' that are either None or empty.
            :param example: A dictionary containing an 'aspects' key with a list of aspects.
            :return: True if any aspect has empty or None 'opinion_words', False otherwise.
            """
            # return any(('opinion_words' in aspect and (aspect['opinion_words'] is None or len(aspect['opinion_words']) ==0)) for aspect in example['aspects'])
            return any(not aspect.get('opinion_words') for aspect in example['aspects'])

        def is_mams(example):
            """
            Determine if there is more than one unique polarity in the aspects of the example.
            :param example: A dictionary containing an 'aspects' key with a list of aspects.
            :return: True if there are multiple unique polarities, False otherwise.
            """
            return len(set(aspect['polarity'] for aspect in example['aspects'])) > 1


        if self.test_type == 'implicit':
            test_examples = [example for example in test_examples if is_implicit(example)]

        elif self.test_type == 'mams':
            test_examples = [example for example in test_examples if is_mams(example)]


        self.raw_datasets = {
            'train': train_examples, 
            'dev'  : dev_examples,
            'test' : test_examples
        }

        print('-----------data statistic-------------')
        for mode in ('train', 'dev', 'test'):
            num_sentences = len(self.raw_datasets[mode])
            num_aspects = sum(len([aspect for aspect in example['aspects'] if aspect['target'] not in ('NULL', None)])
                              for example in self.raw_datasets[mode])

            print(f'{mode.upper():<5} | Sentences: {num_sentences:<5} | Aspect terms: {num_aspects:<5}')

        print('--------------------------------------')

    def prepare_data(self):
        self.load_dataset()

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=8,
            num_workers=1,
            collate_fn=DataCollator(
                tokenizer=self.tokenizer, 
                max_seq_length=self.max_seq_length,
                mode=mode,
                dataset=self.dataset
            )
        )

        print('dataloader-'+mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)



class DataCollator:
    def __init__(self, tokenizer, max_seq_length, mode, dataset):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.dataset = dataset

    def tok(self, text, max_seq_length):
        kwargs = {
            'text': text,
            'return_tensors': 'pt',
        }

        if max_seq_length in (-1, 'longest'):
            kwargs['padding'] = True

        else:
            kwargs['max_length'] = max_seq_length
            kwargs['padding'] = 'max_length'
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)
        return batch_encodings    
    
    def __call__(self, examples):
        text = [example['sentence'] for example in examples]
        batch_encodings = self.tok(text, -1)

        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        labels = None
        if self.mode in ('train', 'dev', 'test'):
            labels = self.make_labels(examples)

        if self.max_seq_length > 0:
            input_ids = input_ids[:, :self.max_seq_length]
            attention_mask = attention_mask[:, :self.max_seq_length]
            labels = labels[:, :self.max_seq_length]

        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'labels'        : labels,
            'examples'      : examples
        }

    def make_labels(self, examples):
        target_seqs = []
        for example in examples:
            target_seq = self.make_atsa_seq(example)
            target_seqs.append(target_seq)

        batch_encodings = self.tok(target_seqs, -1)
        labels = batch_encodings['input_ids']
        labels = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100)
             for l in label]
            for label in labels
        ])

        return labels

    def make_atsa_seq(self, example):
        if 'atsa_seq' in example:
            return example['atsa_seq']

        atsa_seq = []
        for aspect in example['aspects']:
            polarity = aspect['polarity']
            aspect_term = aspect['target']

            if aspect_term in ('NULL', None):
                continue

            assert aspect_term in example['sentence']
            loc = (ifnt0(aspect['from']), ifnt0(aspect['to']))

            atsa_seq.append((aspect_term, polarity, loc))

        atsa_seq = list(set(atsa_seq))
        atsa_seq = sorted(atsa_seq, key=lambda it: it[-1])
        atsa_seq = [f'{aspect_term} | {polarity}' for aspect_term, polarity, _ in atsa_seq]
        atsa_seq = ' ; '.join(atsa_seq)
        example['atsa_seq'] = atsa_seq

        return atsa_seq



def ifnt0(x):
    """
    Return 0 if the input is None, otherwise return the input.
    :param x: The input value that may be None.
    :return: 0 if x is None, x otherwise.
    """
    return 0 if x is None else x
        



class LightningModule(pl.LightningModule):
    def __init__(
        self, 
        learning_rate: float=2e-4,
        adam_epsilon: float=1e-6,
        weight_decay: float=0.,
        warmup_steps: int=0,
        seed: int=42,
        dataset: str='',
        output_dir: str='',
        subname: str='',
        model_name_or_path: str='',
        test_type: str = 'normal', # ('normal', 'implicit', 'multi-aspect')
    ):

        super().__init__()

        self.learning_rate = learning_rate
        self.adam_epsilon  = adam_epsilon
        self.weight_decay  = weight_decay
        self.warmup_steps  = warmup_steps
        self.seed          = seed
        self.dataset       = dataset
        self.output_dir    = output_dir
        self.subname       = subname
        self.test_type     = test_type

        self.model_name_or_path = model_name_or_path
        if self.model_name_or_path in ('', 'subname'):
            self.model_name_or_path = load_json('bash/model_name.json')[self.subname]

        print('model path:', self.model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        print('---------------------------------------------')
        print(self.model_name_or_path)
        print('total params_count:', params_count(self.model))
        # print(self.model.config)
        print('---------------------------------------------')

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def save_model(self):
        dir_name = os.path.join(self.output_dir, 'model', f'dataset={self.dataset},b={self.subname},seed={self.seed}')
        print(f'## save model to {dir_name}')
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.save_pretrained(dir_name)
        self.tokenizer.save_pretrained(dir_name)

    def forward(self, **inputs):
        examples = inputs.pop('examples')
        output   = self.model(**inputs)
        return {'loss': output[0]}

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def eval_step(self, batch, batch_idx):
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_beams=1,
            num_return_sequences=1,
        )
        generateds = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return {
            'examples': batch['examples'],
            'predictions': generateds
        }

    def validation_step(self, batch, batch_idx):
        output = self.eval_step(batch, batch_idx)
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self):
        self.current_val_result = Result.parse_from(self.validation_step_outputs)
        self.current_val_result.cal_metric()
    
        self.update_result = False
        if (not hasattr(self, 'best_val_result')) or (self.best_val_result < self.current_val_result):
            self.best_val_result = self.current_val_result
            self.save_model()
            self.update_result = True

        self.validation_step_outputs.clear()

    # def on_train_end(self):
    #     self.save_model()

    def test_step(self, batch, batch_idx):
        output = self.eval_step(batch, batch_idx)
        self.test_step_outputs.append(output)

    def on_test_epoch_end(self):
        self.test_result = Result.parse_from(self.test_step_outputs)
        self.test_result.cal_metric()
        self.test_result.save_metric(
            self.output_dir, 
            # TODO: 记录原始模型的地址
            self.model_name_or_path, 
            self.subname, 
            self.dataset + self.test_type, 
            self.seed,
            self.learning_rate,
        )
        self.test_result.save_prediction(
            self.output_dir, 
            self.model_name_or_path, 
            self.subname, 
            self.dataset + self.test_type, 
            self.seed,
            self.learning_rate,
        )
        self.test_step_outputs.clear()

    def configure_optimizers(self):

        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon, 
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]



class LoggingCallback(pl.Callback):
    def __init__(self, argument_parser, name_space):
        super().__init__()
        self.argument_parser = argument_parser
        self.name_space      = name_space

    def on_validation_end(self, trainer, pl_module):
        if not pl_module.update_result:
            return 

        if hasattr(pl_module, 'current_train_result'):
            pl_module.current_train_result.report()
        print('------------------------------------------------------------')
        print('[current]', end=' ')
        pl_module.current_val_result.report()

        print('[best]   ', end=' ')
        pl_module.best_val_result.report()
        print('------------------------------------------------------------\n')

    def on_test_end(self, trainer, pl_module):
        pl_module.test_result.report()



def cli_main():
    
    # no effect
    # torch.set_float32_matmul_precision('medium')

    from lightning.pytorch.cli import LightningCLI
    cli = LightningCLI(LightningModule, DataModule, LoggingCallback)


if __name__ == '__main__':
    cli_main()