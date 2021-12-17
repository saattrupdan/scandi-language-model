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

        # Disable tokenizer parallelization
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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

                # Reshape the predictions and labels to shape
                # (batch_size x seq_len, vocab_size) and
                # (batch_size x seq_len,), respectively
                logits = logits.view(-1, logits.shape[-1])
                labels = samples['labels'].view(-1)

                # Compute loss
                loss = F.cross_entropy(logits, labels)
                test_loss += float(loss)

                # Compute accuracy
                predictions = logits.softmax(dim=-1)[labels >= 0]
                labels = labels[labels >= 0]
                metric(predictions, labels)

            # Update progress bar
            pbar.update(batch_size)

        # Close progress bar
        pbar.close()

        # Compute the average loss
        test_loss /= (len(test_dataset) // batch_size)

        # Compute the perplexity
        perplexity = torch.exp(torch.tensor(test_loss))
        print(f'Perplexity: {perplexity}')

        # Compute the average accuracy
        accuracy = metric.compute()
        print(f'Accuracy: {100 * accuracy:.2f}%')

        answer = input('Do you want to push the model to the hub? [y/n]\n> ')

        # Push the tokeniser and model to the hub
        if answer.lower() == 'y':
            tokeniser.push_to_hub(model_id)
            model.push_to_hub(model_id)


if __name__ == '__main__':
    main(batch_size=8)
