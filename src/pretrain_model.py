'''Pretrain a language model on a corpus of text'''

from transformers import (RobertaConfig, RobertaForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, PreTrainedTokenizerFast)
from datasets import Dataset


def main():
    '''Main function'''

    # Load pretrained tokenizer
    tokeniser = PreTrainedTokenizerFast(tokenizer_file='wiki-da-sv-no.json',
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
    dataset = Dataset.load_from_disk('data/dataset')

    # Split dataset into train and validation
    splits = dataset.train_test_split(train_size=0.9, seed=4242)
    train_dataset = splits['train']
    splits = splits['test'].train_test_split(train_size=0.5, seed=4242)
    val_dataset = splits['train']
    test_dataset = splits['test']

    # Tokenise the datasets
    def tokenise(examples: dict) -> dict:
        return tokeniser(examples['text'],
                         truncation=True,
                         padding=True,
                         max_length=512)
    train_dataset = train_dataset.map(tokenise, batched=True)
    val_dataset = val_dataset.map(tokenise, batched=True)
    test_dataset = test_dataset.map(tokenise, batched=True)

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokeniser,
                                                    mlm=True,
                                                    mlm_probability=0.15)

    # Set up training arguments
    training_args = TrainingArguments(output_dir='roberta-base-wiki-da-sv-no',
                                      overwrite_output_dir=True,
                                      num_train_epochs=3,
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=8,
                                      gradient_accumulation_steps=32,
                                      save_steps=100,
                                      save_total_limit=3)

    # Initialise trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset)

    # Train model
    trainer.train()

    # Evaluate model
    trainer.evaluate(test_dataset)

    # Save model
    trainer.save_model()


if __name__ == '__main__':
    main()
