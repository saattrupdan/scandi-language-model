'''Pretrain a language model on a corpus of text'''

from transformers import (RobertaConfig, RobertaForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, PreTrainedTokenizerFast,
                          EarlyStoppingCallback)
from datasets import Dataset, load_metric
import datasets
from functools import partial
import torch
from typing import Dict


datasets.set_caching_enabled(False)


def main(config: dict):
    '''Main function'''

    # Load pretrained tokenizer
    tokeniser = PreTrainedTokenizerFast(tokenizer_file='wiki-da.json',
                                        bos_token='<s>',
                                        cls_token='<s>',
                                        eos_token='</s>',
                                        sep_token='</s>',
                                        unk_token='<unk>',
                                        mask_token='<mask>',
                                        pad_token='<pad>')

    # Set the maximal sequence length
    tokeniser.model_max_length = 512

    # Initialise config
    config = RobertaConfig(pad_token_id=4,
                           bos_token_id=0,
                           eos_token_id=1,
                           vocab_size=len(tokeniser))

    # Initialise model
    model = RobertaForMaskedLM(config=config)

    # Load dataset
    dataset = Dataset.load_from_disk('data/da_dataset')

    # Split dataset into train and validation
    splits = dataset.train_test_split(train_size=0.9,
                                      seed=config['random_seed'])
    train_dataset = splits['train']
    splits = splits['test'].train_test_split(train_size=0.5,
                                             seed=config['random_seed'])
    val_dataset = splits['train']
    test_dataset = splits['test']

    # Tokenise the 128-length dataset
    def tokenise(examples: dict, max_length: int) -> dict:
        return tokeniser(examples['text'],
                         truncation=True,
                         padding=True,
                         max_length=max_length)
    tokenise_128 = partial(tokenise, max_length=128)
    train_dataset_128 = train_dataset.map(tokenise_128, batched=True)
    val_dataset_128 = val_dataset.map(tokenise_128, batched=True)

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokeniser,
                                                    mlm=True,
                                                    mlm_probability=0.15)

    # Count the number of GPUs available, and set the gradient accumulation
    # accordingly, to ensure that the effective batch size is 256
    device_count = torch.cuda.device_count()
    batch_size = config['batch_size_128']
    acc_steps = 300 // (batch_size * device_count)

    # Set up training arguments
    training_args = TrainingArguments(output_dir='roberta-base-wiki-da',
                                      overwrite_output_dir=True,
                                      evaluation_strategy='steps',
                                      logging_strategy='steps',
                                      save_strategy='steps',
                                      eval_steps=500,
                                      logging_steps=500,
                                      save_steps=500,
                                      max_steps=config['num_steps_128'],
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      gradient_accumulation_steps=acc_steps,
                                      metric_for_best_model='accuracy',
                                      save_total_limit=1,
                                      learning_rate=config['lr'],
                                      warmup_steps=config['warmup_steps'],
                                      weight_decay=config['weight_decay'],
                                      report_to='all',
                                      load_best_model_at_end=True)

    # Initialise trainer
    params = dict(early_stopping_patience=config['patience'])
    early_stopping = EarlyStoppingCallback(**params)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics,
                      train_dataset=train_dataset_128.shuffle(),
                      eval_dataset=val_dataset_128,
                      callbacks=[early_stopping])

    # Temp
    trainer.train_dataset = train_dataset_128.shuffle()
    trainer.eval_dataset = val_dataset_128
    trainer.args.max_steps = config['max_steps_128']
    trainer.args.batch_size = config['batch_size_128']
    acc_steps = 300 // (config['batch_size_128'] * device_count)
    trainer.args.gradient_accumulation_steps = acc_steps

    # Train model on 128-length sequences
    trainer.train()

    # Tokenise the 512-length dataset and remove the 128-length datasets
    del train_dataset_128
    del val_dataset_128
    tokenise_512 = partial(tokenise, max_length=512)
    train_dataset_512 = train_dataset.map(tokenise_512, batched=True)
    val_dataset_512 = val_dataset.map(tokenise_512, batched=True)
    test_dataset_512 = test_dataset.map(tokenise_512, batched=True)

    # Set up trainer for 512-length sequence
    trainer.train_dataset = train_dataset_512.shuffle()
    trainer.eval_dataset = val_dataset_512
    trainer.args.max_steps = config['max_steps_512']
    trainer.args.batch_size = config['batch_size_512']
    acc_steps = 300 // (config['batch_size_512'] * device_count)
    trainer.args.gradient_accumulation_steps = acc_steps

    # Train model on 512-length sequences
    trainer.train(resume_from_checkpoint=True)

    # Evaluate model
    trainer.evaluate(test_dataset_512)

    # Save model
    trainer.save_model()


def compute_metrics(predictions_and_labels: tuple) -> Dict[str, float]:
    '''Compute the metrics for MLM evaluation.

    Args:
        predictions_and_labels (pair of arrays):
            The first array contains the probability predictions and the
            second array contains the true labels.

    Returns:
        dict:
            A dictionary with the names of the metrics as keys and the
            metric values as values.
    '''
    # Get predictions and labels
    predictions, labels = predictions_and_labels

    # Compute the accuracy
    rounded_predictions = predictions.argmax(axis=-1)
    acc_metric = load_metric('accuracy')
    acc_results = acc_metric.compute(predictions=rounded_predictions,
                                     references=labels)

    # Compute the perplexity
    bce_loss = torch.nn.BCEWithLogitsLoss()
    bce = bce_loss(predictions, labels)
    perplexity = torch.exp(bce)

    return dict(accuracy=acc_results['accuracy'], perplexity=perplexity)


if __name__ == '__main__':
    config = dict(lr=1e-4,
                  weight_decay=0.01,
                  warmup_steps=10_000,
                  num_steps_128=900_000,
                  num_steps_512=100_000,
                  batch_size_128=32,
                  batch_size_512=8,
                  random_seed=4242)
    main()
