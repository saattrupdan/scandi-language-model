'''Script for training a tokeniser'''

from tokenizers import (normalizers, pre_tokenizers, tokenizers,
                        processors, trainers, models, decoders, AddedToken)
from datasets import Dataset


def main():
    '''Main function'''

    # Load the dataset
    dataset = Dataset.load_from_disk('data/dataset')

    # Load the dictionary
    # csv_path = 'data/retskrivningsordbog-basic.csv'
    # dictionary = pd.read_csv(csv_path, sep=';', names=['word', 'type'])
    # dictionary['type'] = dictionary.type.str.split(',')
    # dictionary = (dictionary
    #               .explode('type')
    #               .query('type in ["sb.", "talord", "sb. pl."]')
    #               .word
    #               .str.replace('[0-9]+[.] ', '', regex=True)
    #               .str.lower()
    #               .drop_duplicates()
    #               .sort_values(key=lambda x: x.str.len())
    #               .tolist())
    # dictionary = [word for word in dictionary
    #               if ' ' not in word and len(word) > 2]

    # Initialise the tokeniser model
    model = models.Unigram()

    # Initialise the tokeniser
    tokeniser = tokenizers.Tokenizer(model=model)

    # Initialise the special tokens
    special_tokens = [
        AddedToken('<s>', single_word=True, normalized=False),
        AddedToken('</s>', single_word=True, normalized=False),
        AddedToken('<unk>', single_word=True, normalized=False),
        AddedToken('<mask>', single_word=True, normalized=False),
        AddedToken('<pad>', single_word=True, normalized=False),
    ]
    tokeniser.add_special_tokens(special_tokens)

    # Initialise the normaliser
    normaliser = normalizers.Sequence([
        normalizers.NFKC()
    ])
    tokeniser.normalizer = normaliser

    # Initialise the pre-tokeniser
    pre_tokeniser = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(add_prefix_space=True),
        pre_tokenizers.Punctuation(behaviour='isolated'),
        pre_tokenizers.Digits(individual_digits=True)
    ])
    tokeniser.pre_tokenizer = pre_tokeniser

    # Initialise the post-processor
    post_processor = processors.RobertaProcessing(cls=('<s>', 0),
                                                  sep=('</s>', 1))
    tokeniser.post_processor = post_processor

    # Initialise the decoder
    decoder = decoders.Metaspace(add_prefix_space=True)
    tokeniser.decoder = decoder

    # Initialise the trainer
    trainer = trainers.UnigramTrainer(vocab_size=200_000,
                                      special_tokens=special_tokens)

    # Train the tokeniser
    tokeniser.train_from_iterator(iterator=dataset['text'], trainer=trainer)

    # Save the tokeniser
    tokeniser.save('dasvno-wiki.json')


if __name__ == '__main__':
    main()

    tokeniser = tokenizers.Tokenizer.from_file('dasvno-wiki.json')

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
        'rosenbusk',
        'tornebusk',
        'tjørnebusk',
        'kapersbusk',
        'ribsbusk'
    ]
    for example in test_examples:
        print(example, tokeniser.encode(example).tokens)
        print()

    while True:
        input_text = input('Enter text:\n> ')
        print(tokeniser.encode(input_text).tokens)
        print()
