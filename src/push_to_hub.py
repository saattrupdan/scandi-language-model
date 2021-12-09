'''Push the model the HF Hub'''

from transformers import (AutoModelForPreTraining, PreTrainedTokenizerFast,
                          DataCollatorForLanguageModeling)
from datasets import Dataset
import sys
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import torchmetrics as tm


def main(batch_size: int):
    '''Push the model to the HF Hub'''
    if len(sys.argv) == 3:

        # Fetch arguments
        model_id = sys.argv[1]
        tokenizer_id = sys.argv[2]

        # Set up metric
        metric = tm.Accuracy().cuda()

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
        model.cuda()

        # Load test dataset
        dataset = Dataset.load_from_disk('data/da_dataset')
        splits = dataset.train_test_split(train_size=0.99,
                                          seed=4242)
        test_dataset = splits['test']

        # Preprocess the test dataset
        def preprocess(examples: dict) -> dict:
            examples = tokeniser(examples['text'],
                                 truncation=True,
                                 padding='max_length',
                                 max_length=512)
            examples = data_collator((examples,), return_tensors='pt')
            return examples
        test_dataset = test_dataset.map(preprocess)

        # Evaluate the model on the test dataset
        test_loss = 0
        pbar = tqdm(total=len(test_dataset), desc='Evaluating')
        for i in range(0, len(test_dataset), batch_size):

            # Get test sample
            samples = test_dataset[i:i+batch_size]

            # Remove the 'text' key from the sample
            samples.pop('text')

            # Convert samples to tensors
            samples = {key: torch.tensor(val).squeeze().cuda()
                       for key, val in samples.items()}

            # Compute metrics
            with torch.no_grad():

                # Get predictions
                logits = model(**samples).logits
                logits = logits[samples['labels'] >= 0]
                labels = samples['labels'][samples['labels'] >= 0]

                # Compute loss
                one_hotted = (F.one_hot(labels, num_classes=logits.shape[-1])
                               .float())
                loss = F.binary_cross_entropy_with_logits(logits, one_hotted)
                test_loss += float(loss)

                # Compute accuracy
                metric(logits.softmax(dim=-1), labels)

            # Update progress bar
            pbar.update(batch_size)

        # Close progress bar
        pbar.close()

        # Compute the average loss
        test_loss /= len(test_dataset)

        # Compute the perplexity
        perplexity = torch.exp(test_loss)
        print(f'Perplexity: {perplexity}')

        # Compute the average accuracy
        accuracy = metric.compute()
        print(f'Accuracy: {100 * accuracy:.2f}%')

        # Push the tokeniser and model to the hub
        tokeniser.push_to_hub(model_id)
        model.push_to_hub(model_id)


if __name__ == '__main__':
    main(batch_size=8)
