# TODO： mixtral
import os
import numpy as np

import torch
import lightning as pl

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from datasets import load_dataset

from utils import mkdir_if_not_exist, yield_data_file



class PretrainingDataModuleMixtral(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str="",
        max_seq_length: int=-1,
        max_seq_length_output: int=-1,
        train_batch_size: int=32,
        eval_batch_size: int=32,
        base_data_dir: str='',
        data_dirs: str='',
        num_workers: int=1,
        cache_dir: str='',
        # train_size: int=102_400,
        test_size: int=10_240,
        test_file_dir : str=''
    ):
        super().__init__()

        self.model_name_or_path    = model_name_or_path
        self.max_seq_length_output = max_seq_length_output
        self.max_seq_length   = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size  = eval_batch_size

        self.base_data_dir = base_data_dir
        self.data_dirs      = data_dirs.split('__')
        self.num_workers   = num_workers
        self.cache_dir     = cache_dir
        # self.train_size    = train_size
        self.test_file_dir = test_file_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.tokenizer.add_tokens(['[NL]'])
        self.collator_fn = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        print('Using MixtralDataModule')
        print('-'*100)

    def prepare_data(self):
        self.load_dataset()
        self.prepare_dataset()

    def load_dataset(self):
        data_files = []
        print('-'*100)
        print('Training Data file used in pre-training：')
        for data_dir in self.data_dirs:
            data_dir = os.path.join(self.base_data_dir, data_dir)
            print(data_dir)
            if os.path.isdir(data_dir):
                data_files.extend(yield_data_file(data_dir))
            else:
                data_files.append(data_dir)
        print('-' * 100)

        kwargs_for_train = {
            'path': 'json',
            'data_files': data_files,
            'verification_mode': 'no_checks'
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'load', 'processed_train.arrow')
            print('Cache data file used in pre-training：')
            print(cache_dir)
            print('-' * 100)
            mkdir_if_not_exist(cache_dir)
            kwargs_for_train['cache_dir'] = cache_dir


        self.raw_train_datasets = load_dataset(**kwargs_for_train)
        print('it is what raw_train_datasets likes:')
        print(self.raw_train_datasets)
        print('-' * 100)

        kwargs_for_test = {
            'path': 'json',
            'data_files': self.test_file_dir,
            'verification_mode': 'no_checks'
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'load', 'processed_test.arrow')
            print('Cache data file used in pre-training：')
            print(cache_dir)
            print('-' * 100)
            mkdir_if_not_exist(cache_dir)
            kwargs_for_test['cache_dir'] = cache_dir

        self.raw_test_datasets = load_dataset(**kwargs_for_test)
        print('it is what raw_test_datasets likes:')
        print(self.raw_test_datasets)
        print('-' * 100)

    def prepare_dataset(self):

        def tokenize_function(examples):
            if 'chatgpt_response' in examples.keys():
                response_type = 'chatgpt_response'
            elif 'mixtral_response' in examples.keys():
                response_type = 'mixtral_response'
            elif 'llama2_response' in examples.keys():
                response_type = 'llama2_response'

            examples['response'] = examples.pop(response_type)
            if 'prompt' in examples:
                batch_encodings = self.tokenizer(
                    [(prompt + input_text).replace('\n', '[NL]') for prompt, input_text in zip(examples['prompt'], examples['Text'])],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length,
                )

                output_batch_encodings = self.tokenizer(
                    [llama_response.replace('\n', '[NL]') for llama_response in examples['extracted_llama_response']],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length_output,
                    return_attention_mask=False,
                )

            else:

                batch_encodings = self.tokenizer(
                    [text.replace('\n', '[NL]') for text in examples['Text']],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length,
                )

                output_batch_encodings = self.tokenizer(
                    [response.replace('\n', '[NL]') for response in examples['response']],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length_output,
                    return_attention_mask=False,
                )



            return {
                'input_ids': batch_encodings['input_ids'],
                'attention_mask': batch_encodings['attention_mask'],
                'decoder_input_ids': output_batch_encodings['input_ids'],
            }


        kwargs_train = {
            'batched': True,
            'remove_columns': ['ID', 'Overall','mixtral_response'],
            'num_proc': 64,
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'tokenize', 'processed_train.arrow')
            mkdir_if_not_exist(cache_dir)
            kwargs_train['load_from_cache_file'] = True
            kwargs_train['cache_file_names'] = {'train': cache_dir}


        processed_train_datasets = self.raw_train_datasets.map(tokenize_function, **kwargs_train)
        processed_train_datasets = processed_train_datasets.with_format("numpy")

        self.train_dataset = processed_train_datasets['train']
        print("it's the train_dataset like :")
        print(self.train_dataset)

        kwargs_test = {
            'batched': True,
            'remove_columns': ['ID', 'Overall','mixtral_response'],
            'num_proc': 64,
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'tokenize', 'processed_test.arrow')
            mkdir_if_not_exist(cache_dir)
            kwargs_test['load_from_cache_file'] = True
            kwargs_test['cache_file_names'] = {'train': cache_dir}

        processed_test_datasets = self.raw_test_datasets.map(tokenize_function, **kwargs_test)
        processed_test_datasets = processed_test_datasets.with_format("numpy")
        print('-' * 100)
        self.eval_dataset = processed_test_datasets['train']
        print("it's the test_dataset like :")
        print(self.eval_dataset)
        print('-' * 100)


    def get_dataloader(self, mode, batch_size, shuffle):
        dataset = self.train_dataset if mode == 'train' else self.eval_dataset
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
            pin_memory=True,
            prefetch_factor=16
        )

        print(mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

class PretrainingDataModuleLlama(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str="",
        max_seq_length: int=-1,
        max_seq_length_output: int=-1,
        train_batch_size: int=32,
        eval_batch_size: int=32,
        base_data_dir: str='',
        data_dirs: str='',
        num_workers: int=1,
        cache_dir: str='',
        # train_size: int=102_400,
        test_size: int=10_240,
        test_file_dir : str=''
    ):
        super().__init__()

        self.model_name_or_path    = model_name_or_path
        self.max_seq_length_output = max_seq_length_output
        self.max_seq_length   = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size  = eval_batch_size

        self.base_data_dir = base_data_dir
        self.data_dirs      = data_dirs.split('__')
        self.num_workers   = num_workers
        self.cache_dir     = cache_dir
        # self.train_size    = train_size
        self.test_file_dir = test_file_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.tokenizer.add_tokens(['[NL]'])
        self.collator_fn = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        print('Using LlamaDataModule')
        print('-'*100)

    def prepare_data(self):
        self.load_dataset()
        self.prepare_dataset()

    def load_dataset(self):
        data_files = []
        print('-'*100)
        print('Training Data file used in pre-training：')
        for data_dir in self.data_dirs:
            data_dir = os.path.join(self.base_data_dir, data_dir)
            print(data_dir)
            if os.path.isdir(data_dir):
                data_files.extend(yield_data_file(data_dir))
            else:
                data_files.append(data_dir)
        print('-' * 100)

        kwargs_for_train = {
            'path': 'json',
            'data_files': data_files,
            'verification_mode': 'no_checks'
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'load', 'processed_train.arrow')
            print('Cache data file used in pre-training：')
            print(cache_dir)
            print('-' * 100)
            mkdir_if_not_exist(cache_dir)
            kwargs_for_train['cache_dir'] = cache_dir


        self.raw_train_datasets = load_dataset(**kwargs_for_train)
        print('it is what raw_train_datasets likes:')
        print(self.raw_train_datasets)
        print('-' * 100)

        kwargs_for_test = {
            'path': 'json',
            'data_files': self.test_file_dir,
            'verification_mode': 'no_checks'
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'load', 'processed_test.arrow')
            print('Cache data file used in pre-training：')
            print(cache_dir)
            print('-' * 100)
            mkdir_if_not_exist(cache_dir)
            kwargs_for_test['cache_dir'] = cache_dir

        self.raw_test_datasets = load_dataset(**kwargs_for_test)
        print('it is what raw_test_datasets likes:')
        print(self.raw_test_datasets)
        print('-' * 100)

    def prepare_dataset(self):

        def tokenize_function(examples):
            if 'chatgpt_response' in examples.keys():
                response_type = 'chatgpt_response'
            elif 'mixtral_response' in examples.keys():
                response_type = 'mixtral_response'
            elif 'llama2_response' in examples.keys():
                response_type = 'llama2_response'

            examples['response'] = examples.pop(response_type)
            if 'prompt' in examples:
                batch_encodings = self.tokenizer(
                    [(prompt + input_text).replace('\n', '[NL]') for prompt, input_text in zip(examples['prompt'], examples['Text'])],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length,
                )

                output_batch_encodings = self.tokenizer(
                    [llama_response.replace('\n', '[NL]') for llama_response in examples['extracted_llama_response']],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length_output,
                    return_attention_mask=False,
                )

            else:

                batch_encodings = self.tokenizer(
                    [text.replace('\n', '[NL]') for text in examples['Text']],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length,
                )

                output_batch_encodings = self.tokenizer(
                    [response.replace('\n', '[NL]') for response in examples['response']],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length_output,
                    return_attention_mask=False,
                )



            return {
                'input_ids': batch_encodings['input_ids'],
                'attention_mask': batch_encodings['attention_mask'],
                'decoder_input_ids': output_batch_encodings['input_ids'],
            }


        kwargs_train = {
            'batched': True,
            'remove_columns': ['ID', 'Overall','llama2_response'],
            'num_proc': 64,
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'tokenize', 'processed_train.arrow')
            mkdir_if_not_exist(cache_dir)
            kwargs_train['load_from_cache_file'] = True
            kwargs_train['cache_file_names'] = {'train': cache_dir}


        processed_train_datasets = self.raw_train_datasets.map(tokenize_function, **kwargs_train)
        processed_train_datasets = processed_train_datasets.with_format("numpy")

        self.train_dataset = processed_train_datasets['train']
        print("it's the train_dataset like :")
        print(self.train_dataset)

        kwargs_test = {
            'batched': True,
            'remove_columns': ['ID', 'Overall','llama2_response'],
            'num_proc': 64,
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'tokenize', 'processed_test.arrow')
            mkdir_if_not_exist(cache_dir)
            kwargs_test['load_from_cache_file'] = True
            kwargs_test['cache_file_names'] = {'train': cache_dir}

        processed_test_datasets = self.raw_test_datasets.map(tokenize_function, **kwargs_test)
        processed_test_datasets = processed_test_datasets.with_format("numpy")
        print('-' * 100)
        self.eval_dataset = processed_test_datasets['train']
        print("it's the test_dataset like :")
        print(self.eval_dataset)
        print('-' * 100)


    def get_dataloader(self, mode, batch_size, shuffle):
        dataset = self.train_dataset if mode == 'train' else self.eval_dataset
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
            pin_memory=True,
            prefetch_factor=16
        )

        print(mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)


class PretrainingDataModuleChatgpt(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str="",
        max_seq_length: int=-1,
        max_seq_length_output: int=-1,
        train_batch_size: int=32,
        eval_batch_size: int=32,
        base_data_dir: str='',
        data_dirs: str='',
        num_workers: int=1,
        cache_dir: str='',
        # train_size: int=102_400,
        test_file_dir: str='',
        test_size: int=10_240,
    ):
        super().__init__()

        self.model_name_or_path    = model_name_or_path
        self.max_seq_length_output = max_seq_length_output
        self.max_seq_length   = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size  = eval_batch_size

        self.base_data_dir = base_data_dir
        self.data_dirs      = data_dirs.split('__')
        self.num_workers   = num_workers
        self.cache_dir     = cache_dir
        # self.train_size    = train_size
        self.test_size     = test_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.tokenizer.add_tokens(['[NL]'])
        self.collator_fn = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        print('Using ChatGPTDataModule')
        print('-'*100)

    def prepare_data(self):
        self.load_dataset()
        self.prepare_dataset()

    def load_dataset(self):
        data_files = []
        for data_dir in self.data_dirs:
            data_dir = os.path.join(self.base_data_dir, data_dir)
            print(data_dir)
            if os.path.isdir(data_dir):
                data_files.extend(yield_data_file(data_dir))
            else:
                data_files.append(data_dir)

        kwargs = {
            'path': 'json',
            'data_files': data_files,
            'verification_mode': 'no_checks'
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'load', 'processed.arrow')
            print(cache_dir)
            mkdir_if_not_exist(cache_dir)
            kwargs['cache_dir'] = cache_dir

        self.raw_datasets = load_dataset(**kwargs)
        print('it is raw dataset like :')
        print(self.raw_datasets)
        print('-'*100)

    def prepare_dataset(self):

        def tokenize_function(examples):

            if 'prompt' in examples:
                batch_encodings = self.tokenizer(
                    [(prompt + input_text).replace('\n', '[NL]') for prompt, input_text in
                     zip(examples['prompt'], examples['input_text'])],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length,
                )

                output_batch_encodings = self.tokenizer(
                    [output_text.replace('\n', '[NL]') for output_text in examples['output_text']],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length_output,
                    return_attention_mask=False,
                )

            else:

                batch_encodings = self.tokenizer(
                    [text.replace('\n', '[NL]') for text in examples['Text']],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length,
                )

                output_batch_encodings = self.tokenizer(
                    [chatgpt_response.replace('\n', '[NL]') for chatgpt_response in examples['chatgpt_response']],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length_output,
                    return_attention_mask=False,
                )

            return {
                'input_ids': batch_encodings['input_ids'],
                'attention_mask': batch_encodings['attention_mask'],
                'decoder_input_ids': output_batch_encodings['input_ids'],
            }

        kwargs = {
            'batched': True,
            'remove_columns': ['ID', 'Overall','s_starts','s_ends','a_starts','a_ends'],
            # 'remove_columns': ['ID', 'Text', 'chatgpt_response', 'Overall'],
            # 'remove_columns': ['ID', 'input_text', 'prompt', 'output_text', 'Overall'],
            'num_proc': 64,
        }
        if self.cache_dir:
            cache_dir = os.path.join(self.cache_dir, 'tokenize', 'processed.arrow')
            mkdir_if_not_exist(cache_dir)
            kwargs['load_from_cache_file'] = True
            kwargs['cache_file_names'] = {'train': cache_dir}

        processed_datasets = self.raw_datasets.map(tokenize_function, **kwargs)

        print(processed_datasets)

        processed_datasets = processed_datasets['train'].train_test_split(
            test_size=self.test_size,
            seed=42,
            # train_size=self.train_size
        )
        print(processed_datasets)

        # https://discuss.huggingface.co/t/solved-image-dataset-seems-slow-for-larger-image-size/10960/6
        processed_datasets = processed_datasets.with_format("numpy")
        # dataset.set_format(type='torch', columns=['input_ids'])

        self.train_dataset = processed_datasets['train']
        self.eval_dataset = processed_datasets['test']


    def get_dataloader(self, mode, batch_size, shuffle):
        dataset = self.train_dataset if mode == 'train' else self.eval_dataset
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
            pin_memory=True,
            prefetch_factor=16
        )

        print(mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        # print('it is what datacollator receive from dataloader batch examples:')
        # print(examples)


        input_ids      = torch.tensor(np.array([example['input_ids'] for example in examples]), dtype=torch.long)
        attention_mask = torch.tensor(np.array([example['attention_mask'] for example in examples]), dtype=torch.long)

        labels = torch.tensor(np.array([example['decoder_input_ids'] for example in examples]), dtype=torch.long)
        labels[labels==self.tokenizer.pad_token_id] = -100

        input_ids, attention_mask = self.remove_pad(input_ids, attention_mask)
        labels = self.remove_pad_labels(labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def remove_pad(self, input_ids, attention_mask):
        lengths = attention_mask.sum(dim=-1)
        max_length = lengths.max()
        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length]
        return input_ids, attention_mask

    def remove_pad_labels(self, labels):
        mask = (labels!=-100)
        labels, _ = self.remove_pad(labels, mask)
        return labels