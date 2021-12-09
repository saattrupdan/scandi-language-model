'''Push the model the HF Hub'''

from transformers import (AutoModelForPreTraining, PreTrainedTokenizerFast,
                          DataCollatorForLanguageModeling)
from datasets import Dataset
import sys
import torch
from tqdm.auto import trange


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

        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokeniser,
                                                        mlm=True,
                                                        mlm_probability=0.15)

        # Load pretrained model
        model = AutoModelForPreTraining.from_pretrained(model_id)
        model.eval()
        model.cpu()

        # Load test dataset
        dataset = Dataset.load_from_disk('data/da_dataset')
        splits = dataset.train_test_split(train_size=0.99,
                                          seed=4242)
        test_dataset = splits['test']

        # Preprocess the test dataset
        def preprocess(examples: dict) -> dict:
            examples = tokeniser(examples['text'],
                                 truncation=True,
                                 padding=True,
                                 max_length=512)
            examples = data_collator(examples)
            return examples
        test_dataset = test_dataset.map(preprocess)

        # Evaluate the model on the test dataset
        test_loss = 0
        for i in trange(0, len(test_dataset), 8):

            # Get test sample
            samples = test_dataset[i:i+8]

            # Remove the 'text' key from the sample
            samples.pop('text')

            # Get loss
            with torch.no_grad():
                test_loss += model(**samples).loss

        # Compute the average loss
        test_loss /= len(test_dataset)

        # Compute the perplexity
        perplexity = torch.exp(test_loss)
        print(f'Perplexity: {perplexity}')

        # Push the tokeniser and model to the hub
        tokeniser.push_to_hub(model_id)
        model.push_to_hub(model_id)


if __name__ == '__main__':
    main()
