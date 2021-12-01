'''Script for training a tokeniser'''

from tokenizers import (normalizers, pre_tokenizers, tokenizers,
                        processors, trainers, models, AddedToken)
from datasets import Dataset
import pandas as pd

from compound_nouns import CompoundNounPreTokenizer


def main():
    '''Main function'''

    # Load the dataset
    dataset = Dataset.load_from_disk('data/dataset')

    # Initialise the tokeniser model
    model = models.WordPiece(unk_token='<unk>')

    # Initialise the tokeniser
    tokeniser = tokenizers.Tokenizer(model=model)

    # Initialise the special tokens
    special_tokens = [
        AddedToken('<s>', single_word=True, normalized=False),
        AddedToken('</s>', single_word=True, normalized=False),
        AddedToken('<unk>', single_word=True, normalized=False),
        AddedToken('<mask>', single_word=True, normalized=False),
        AddedToken('<pad>', single_word=True, normalized=False)
    ]
    tokeniser.add_special_tokens(special_tokens)

    # Initialise the normaliser
    normaliser = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Strip(),
        normalizers.Lowercase()
    ])
    tokeniser.normalizer = normaliser

    # Load the dictionary
    csv_path = 'data/retskrivningsordbog-basic.csv'
    dictionary = (pd.read_csv(csv_path, sep=';', names=['word', 'type'])
                    .query('type == "sb."')
                    .word
                    .str.replace('[0-9]+[.] ', '', regex=True)
                    .str.lower()
                    .drop_duplicates()
                    .sort_values(key=lambda x: x.str.len())
                    .tolist())
    dictionary = [word for word in dictionary
                  if ' ' not in word and len(word) > 1]

    # Initialise the pre-tokeniser
    compound_noun = CompoundNounPreTokenizer(dictionary=dictionary,
                                             interfixes=['e', 's'])
    compound_noun_pretok = pre_tokenizers.PreTokenizer.custom(compound_noun)
    pre_tokeniser = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(add_prefix_space=False),
        pre_tokenizers.Punctuation(behaviour='isolated'),
        pre_tokenizers.Digits(individual_digits=True),
        compound_noun_pretok
    ])
    tokeniser.pre_tokenizer = pre_tokeniser

    # Initialise the post-processor
    post_processor = processors.RobertaProcessing(cls=('<s>', 0),
                                                  sep=('</s>', 1))
    tokeniser.post_processor = post_processor

    # Initialise the trainer
    trainer = trainers.WordPieceTrainer(vocab_size=50_000,
                                        min_frequency=0,
                                        special_tokens=special_tokens,
                                        continuing_subword_prefix='##')

    # Train the tokeniser
    tokeniser.train_from_iterator(iterator=dataset['text'], trainer=trainer)

    # Print the vocabulary size
    print(f'Vocabulary size: {tokeniser.get_vocab_size():,}')

    test_examples = [
        'brændenældesuppe',
        'cykelanhænger',
        'bronzemedalje',
        'dagsommerfugl',
        'ferskvandsfisk',
        'gerningsstedsundersøgelse',
        'fødselsdagshilsen',
        'fostervandsundersøgelse',
        'havregrynkugle',
        'jernbaneoverskæring',
        'kvælstofudvaskning',
        'ligkistesnedker',
        'ladeplads',
        'lungetransplantation',
        'motorcykelhandske',
        'nyhedsoplæser',
        'nyretransplantation',
        'fiskeørnsæg',
        'fyrværkeriulykke',
        'førstegenerationsindvandrer',
        'arbejdsmarkedsuddannelse',
        'halvfjerdsårsfødselsdag',
    ]
    for example in test_examples:
        print(example, compound_noun_pretok.pre_tokenize_str(example))
        print(example, tokeniser.encode(example).tokens)
        print()

    # Save the tokeniser
    # tokeniser.save('scandiwikibert')


if __name__ == '__main__':
    main()
