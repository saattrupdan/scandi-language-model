'''Push the model the HF Hub'''

from transformers import (AutoModelForPreTraining, PreTrainedTokenizerFast,
                          TrainingArguments, DataCollatorForLanguageModeling,
                          Trainer)
from datasets import Dataset
import sys

from pretrain_model import compute_metrics


def main():
    '''Push the model to the HF Hub'''
    if len(sys.argv) == 3:

        # Fetch arguments
        model_id = sys.argv[1]
        tokenizer_id = sys.argv[2]

        # Ensure that `tokenizer_id` has file suffix
        if not tokenizer_id.endswith('.json'):
            tokenizer_id += '.json'

        # Load pretrained tokenizer and push it to the hub
        tokeniser = PreTrainedTokenizerFast(tokenizer_file=tokenizer_id,
                                            bos_token='<s>',
                                            cls_token='<s>',
                                            eos_token='</s>',
                                            sep_token='</s>',
                                            unk_token='<unk>',
                                            mask_token='<mask>',
                                            pad_token='<pad>')
        tokeniser.model_max_length = 512
        tokeniser.push_to_hub(model_id)

        # Load pretrained model and push it to the hub
        model = AutoModelForPreTraining.from_pretrained(model_id)
        model.eval()

        # Load test dataset
        dataset = Dataset.load_from_disk('data/da_dataset')
        splits = dataset.train_test_split(train_size=0.99,
                                          seed=config['random_seed'])
        test_dataset = splits['test']

        # Tokenise the dataset
        def tokenise(examples: dict) -> dict:
            return tokeniser(examples['text'],
                             truncation=True,
                             padding=True,
                             max_length=512)
        test_dataset = test_dataset.map(tokenise, batched=True)

        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokeniser,
                                                        mlm=True,
                                                        mlm_probability=0.15)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='roberta-base-wiki-da',
            per_device_eval_batch_size=4,
            eval_accumulation_steps=1
        )

        # Initialise trainer
        trainer = Trainer(model=model,
                          args=training_args,
                          data_collator=data_collator,
                          compute_metrics=compute_metrics,
                          eval_dataset=test_dataset)

        # Evaluate model
        trainer.evaluate(test_dataset)

        model.push_to_hub(model_id)


if __name__ == '__main__':
    main()
