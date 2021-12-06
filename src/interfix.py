'''A pre-tokeniser for compound nouns'''

from tokenizers import PreTokenizedString, NormalizedString
from typing import List
import pandas as pd


class InterfixPreTokenizer:
    '''A pre-tokeniser for interfixes.

    Args:
        dictionary (list of str):
            A list of words to be considered as components of compound nouns.
        interfixes (list of str, optional):
            A list of words to be considered as interfixes between components
            of compound nouns. Defaults to ['e', 'n', 's'], being the list of
            Danish interfixes.

    Attributes:
        dictionary (list of str): Components of compound nouns.
        interfixes (list of str): Possible interfixes in compound nouns.
    '''
    def __init__(self,
                 dictionary: List[str],
                 interfixes: List[str] = ['e', 'n', 's']):
        dictionary = pd.DataFrame.from_dict(dict(words=dictionary))
        dictionary['length'] = dictionary.words.str.len()
        dictionary = dictionary.sort_values(by='length')
        self.dictionary = {length: dictionary.query('length == @length')
                                             .words
                                             .tolist()
                           for length in range(1, dictionary.length.max())}
        self.interfixes = interfixes

    def _lookup(self, string: str) -> bool:
        '''Check if the given string is a dictionary word.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the string is a dictionary word, False otherwise.
        '''
        return string in self.dictionary.get(len(string), [])

    def compound_split(self,
                       _: int,
                       normalized_string: NormalizedString
                       ) -> List[NormalizedString]:
        '''Split a normalized string into a list of normalized strings.

        Args:
            normalized_string (NormalizedString):
                The normalized string to split.

        Returns:
            list of NormalizedString objects:
                The list of normalized strings.
        '''
        string = str(normalized_string)
        splits = list()
        potential_ifixes = [char_idx for char_idx, char in enumerate(string)
                            if char in self.interfixes]
        for char_idx in potential_ifixes:

            if (self._lookup(string[:char_idx]) and
                    not self._lookup(string[:char_idx + 1])):

                remaining = normalized_string[char_idx + 1:]
                remaining_splits = self.compound_split(0, remaining)

                if (len(remaining_splits) == 1 and
                        not self._lookup(str(remaining_splits[0]))):
                    continue

                else:
                    splits.append(normalized_string[:char_idx])
                    splits.append(NormalizedString('<interfix>'))
                    splits.extend(remaining_splits)
                    return splits

        else:
            return [normalized_string]

    def pre_tokenize(self, pretok: PreTokenizedString):
        '''Pre-tokenise the given string in-place.

        Args:
            pretok (PreTokenizedString): The string to pre-tokenise.
        '''
        pretok.split(self.compound_split)


if __name__ == '__main__':
    # Load the dictionary
    csv_path = 'data/retskrivningsordbog-basic.csv'
    dictionary = pd.read_csv(csv_path, sep=';', names=['word', 'type'])
    dictionary['type'] = dictionary.type.str.split(',')
    dictionary = (dictionary
                  .explode('type')
                  .query('type in ["sb.", "talord", "sb. pl."]')
                  .word
                  .str.replace('[0-9]+[.] ', '', regex=True)
                  .str.lower()
                  .drop_duplicates()
                  .sort_values(key=lambda x: x.str.len())
                  .tolist())
    dictionary = [word for word in dictionary
                  if ' ' not in word and len(word) > 2]

    # Initialise the pre-tokeniser
    interfix_pretok = InterfixPreTokenizer(dictionary=dictionary,
                                           interfixes=['e', 's', 'n'])

    from transformers import AutoTokenizer
    electra_model_id = 'Maltehb/-l-ctra-danish-electra-small-cased'
    electra_tok = AutoTokenizer.from_pretrained(electra_model_id)
    dabert_model_id = 'Maltehb/danish-bert-botxo'
    dabert_tok = AutoTokenizer.from_pretrained(dabert_model_id)

    # Pre-tokenise the string
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
        'hestesadel',
        'fiskeørn',
    ]
    for example in test_examples:
        norm_example = NormalizedString(example)
        print(example)
        print(list(map(str, interfix_pretok.compound_split(0, norm_example))))
        print([electra_tok.decode(i) for i in electra_tok.encode(example)])
        print([dabert_tok.decode(i) for i in dabert_tok.encode(example)])
        print()
