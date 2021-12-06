'''Script for training a tokeniser'''

from tokenizers import (normalizers, pre_tokenizers, tokenizers,
                        processors, trainers, models, decoders, AddedToken)
from datasets import Dataset


def main():
    '''Main function'''

    # Load the dataset
    dataset = Dataset.load_from_disk('data/da_dataset')

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
    normaliser = normalizers.NFKC()
    tokeniser.normalizer = normaliser

    # Initialise the pre-tokeniser
    pre_tokeniser = pre_tokenizers.Metaspace(add_prefix_space=True)
    tokeniser.pre_tokenizer = pre_tokeniser

    # Initialise the post-processor
    params = dict(cls=('<s>', 0), sep=('</s>', 1))
    post_processor = processors.RobertaProcessing(**params)
    tokeniser.post_processor = post_processor

    # Initialise the decoder
    decoder = decoders.Metaspace(add_prefix_space=True)
    tokeniser.decoder = decoder

    # Initialise the trainer
    trainer = trainers.UnigramTrainer(vocab_size=32_000,
                                      special_tokens=special_tokens,
                                      unk_token='<unk>')

    # Train the tokeniser
    tokeniser.train_from_iterator(iterator=dataset['text'], trainer=trainer)

    # Save the tokeniser
    tokeniser.save('wiki-da.json')


if __name__ == '__main__':
    #main()

    tokeniser = tokenizers.Tokenizer.from_file('wiki-da.json')

    # Load other tokenisers
    from transformers import AutoTokenizer
    electra_model_id = 'Maltehb/-l-ctra-danish-electra-small-cased'
    electra_tok = AutoTokenizer.from_pretrained(electra_model_id)
    dabert_model_id = 'Maltehb/danish-bert-botxo'
    dabert_tok = AutoTokenizer.from_pretrained(dabert_model_id)

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
        print([electra_tok.decode(i) for i in electra_tok.encode(example)])
        print([dabert_tok.decode(i) for i in dabert_tok.encode(example)])
        print()

    while True:
        input_text = input('Enter text:\n> ')
        print(tokeniser.encode(input_text).tokens)
        print([electra_tok.decode(i) for i in electra_tok.encode(input_text)])
        print([dabert_tok.decode(i) for i in dabert_tok.encode(input_text)])
        print()
