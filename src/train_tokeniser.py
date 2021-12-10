'''Script for training a tokeniser'''

from tokenizers import trainers
from datasets import Dataset
import datasets
import pandas as pd
from interfix import InterfixTokeniser


datasets.set_caching_enabled(False)


def main():
    '''Main function'''

    # Load the dataset
    dataset = Dataset.load_from_disk('data/da_dataset')

    # Load the dictionary file and load all the nouns from it
    csv_path = 'data/retskrivningsordbog.csv'
    dictionary = pd.read_csv(csv_path, sep=';', names=['base', 'word', 'type'])
    dictionary.type = dictionary.type.str.split(',')
    dictionary = (dictionary
                  .explode('type')
                  .query('type in ["sb.", "talord", "sb. pl."]'))

    # Remove banned words
    banned_words = ['ren', 'ger']
    dictionary = dictionary[~dictionary.word.isin(banned_words)]

    # Load the small dictionary
    small_dictionary = (dictionary
                        .base
                        .str.replace('[0-9]+[.] ', '', regex=True)
                        .str.lower()
                        .drop_duplicates()
                        .sort_values(key=lambda x: x.str.len())
                        .tolist())
    small_dictionary = [word for word in small_dictionary
                        if ' ' not in word and len(word) > 1]

    # Load the large dictionary
    dictionary = (dictionary
                  .word
                  .str.replace('[0-9]+[.] ', '', regex=True)
                  .str.lower()
                  .drop_duplicates()
                  .sort_values(key=lambda x: x.str.len())
                  .tolist())
    dictionary = [word for word in dictionary
                  if ' ' not in word and len(word) > 1]

    # Initialise the tokeniser
    tokeniser = InterfixTokeniser(small_dictionary=small_dictionary,
                                  dictionary=dictionary,
                                  interfixes=['e', 'n', 's'])

    # Initialise the trainer
    trainer = trainers.UnigramTrainer(vocab_size=50_000,
                                      special_tokens=tokeniser.special_tokens,
                                      unk_token='<unk>')

    # Train the tokeniser
    tokeniser.train_from_iterator(iterator=dataset['text'][:1000],
                                  trainer=trainer)

    # Save the tokeniser
    tokeniser.save('interfix-tokeniser-wiki-da')

    return tokeniser


if __name__ == '__main__':
    main()

    tokeniser = InterfixTokeniser.load('interfix-tokeniser-wiki-da')

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
        'ribsbusk',
        'angrebsvinkel'
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
