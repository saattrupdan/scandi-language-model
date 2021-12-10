'''A tokeniser for interfixes'''

from tokenizers import (PreTokenizedString, NormalizedString, normalizers,
                        pre_tokenizers, tokenizers, processors, models,
                        AddedToken)
from typing import List, Union, Optional
import pandas as pd
from pathlib import Path
import json


class InterfixTokeniser:
    '''A tokeniser for interfixes.

    Args:
        small_dictionary (list of str):
            A list of words to be considered as components of compound nouns,
            aside from the last one in each word.
        dictionary (list of str):
            A list of words to be considered as the final components of
            compound nouns. This dictionary thus contains all the case endings
            as well.
        interfixes (list of str, optional):
            A list of words to be considered as interfixes between components
            of compound nouns. Defaults to ['e', 'n', 's'], being the list of
            Danish interfixes.
        base_tokeniser (tokenizers.Tokenizer, optional):
            A tokeniser to use as a base for the interfix tokeniser. Defaults
            to None, in which case the tokeniser is initialised with a
            Unigram model.

    Attributes:
        small_dictionary (dict): Dictionary containing the small dictionary.
        dictionary (dict): Dictionary containing the dictionary.
        interfixes (list of str): Possible interfixes in compound nouns.
    '''
    def __init__(self,
                 small_dictionary: List[str],
                 dictionary: List[str],
                 interfixes: List[str] = ['e', 'n', 's'],
                 base_tokeniser: Optional[tokenizers.Tokenizer] = None):
        super().__init__()

        if base_tokeniser is None:
            self._tokeniser = tokenizers.Tokenizer(models.Unigram())
        else:
            self._tokeniser = base_tokeniser

        self.interfixes = interfixes

        # Initialise the special tokens
        self.special_tokens = [
            AddedToken('<pad>', single_word=True, normalized=False),
            AddedToken('<s>', single_word=True, normalized=False),
            AddedToken('</s>', single_word=True, normalized=False),
            AddedToken('<unk>', single_word=True, normalized=False),
            AddedToken('<mask>', single_word=True, normalized=False),
            AddedToken('<interfix>', single_word=True, normalized=False),
        ]
        self._tokeniser.add_special_tokens(self.special_tokens)

        # Initialise the normaliser
        normaliser = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Lowercase()
        ])
        self._tokeniser.normalizer = normaliser

        # Initialise the pre-tokeniser
        self.interfix_pretok = InterfixPreTokeniser(
            small_dictionary=small_dictionary,
            dictionary=dictionary,
            interfixes=interfixes
        )
        pre_tokeniser = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.PreTokenizer.custom(self.interfix_pretok),
        ])
        self._tokeniser.pre_tokenizer = pre_tokeniser

        # Initialise the post-processor
        params = dict(cls=('<s>', 1), sep=('</s>', 2))
        post_processor = processors.RobertaProcessing(**params)
        self._tokeniser.post_processor = post_processor

        # Enable truncation and padding
        self._tokeniser.enable_truncation(max_length=512)
        self._tokeniser.enable_padding(pad_id=0,
                                       pad_type_id=0,
                                       pad_token='<pad>')

    def __call__(self, *args, **kwargs):
        return self._tokeniser(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self._tokeniser.encode(*args, **kwargs)

    def encode_batch(self, *args, **kwargs):
        return self._tokeniser.encode_batch(*args, **kwargs)

    def train_from_iterator(self, *args, **kwargs):
        return self._tokeniser.train_from_iterator(*args, **kwargs)

    def get_vocab(self, *args, **kwargs):
        return self._tokeniser.get_vocab(*args, **kwargs)

    def get_vocab_size(self, *args, **kwargs):
        return self._tokeniser.get_vocab_size(*args, **kwargs)

    def id_to_token(self, *args, **kwargs):
        return self._tokeniser.id_to_token(*args, **kwargs)

    def token_to_id(self, *args, **kwargs):
        return self._tokeniser.token_to_id(*args, **kwargs)

    def save(self, directory: Union[str, Path]):
        '''Save the tokeniser to a file.

        Args:
            directory (str or Path): The directory to save the tokeniser to.
        '''
        # Ensure that the directory exists
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir()

        # Save the tokeniser without the interfix pre-tokeniser
        self._tokeniser.pre_tokenizer = pre_tokenizers.Whitespace()
        self._tokeniser.save(str(directory / 'base.json'))

        # Save the small dictionary
        path = directory / 'small_dictionary.json'
        with path.open('w') as f:
            small_dictionary = self.interfix_pretok.small_dictionary
            small_dictionary = [word for lst in small_dictionary.values()
                                for word in lst]
            json.dump(small_dictionary, f)

        # Save the small dictionary
        path = directory / 'dictionary.json'
        with path.open('w') as f:
            dictionary = self.interfix_pretok.dictionary
            dictionary = [word for lst in dictionary.values() for word in lst]
            json.dump(dictionary, f)

        # Save the interfix config
        path = directory / 'interfix_config.json'
        with path.open('w') as f:
            interfix_config = dict(interfixes=self.interfix_pretok.interfixes)
            json.dump(interfix_config, f)

    @classmethod
    def load(cls, directory: Union[str, Path]):
        '''Load the tokeniser from a file.

        Args:
            directory (str or Path): The directory to load the tokeniser from.
        '''
        # Ensure that the directory exists
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f'Directory {directory} does not exist.')

        # Load the base tokeniser
        base_path = directory / 'base.json'
        base_tokeniser = tokenizers.Tokenizer.from_file(str(base_path))

        # Load the small dictionary
        path = directory / 'small_dictionary.json'
        with path.open('r') as f:
            small_dictionary = json.load(f)

        # Load the dictionary
        path = directory / 'dictionary.json'
        with path.open('r') as f:
            dictionary = json.load(f)

        # Load the interfix config
        path = directory / 'interfix_config.json'
        with path.open('r') as f:
            interfix_config = json.load(f)

        # Initialise the tokeniser
        tokeniser = InterfixTokeniser(
            small_dictionary=small_dictionary,
            dictionary=dictionary,
            base_tokeniser = base_tokeniser,
            **interfix_config
        )

        return tokeniser


class InterfixPreTokeniser:
    '''A pre-tokeniser for interfixes.

    Args:
        small_dictionary (list of str):
            A list of words to be considered as components of compound nouns,
            aside from the last one in each word.
        dictionary (list of str):
            A list of words to be considered as the final components of
            compound nouns. This dictionary thus contains all the case endings
            as well.
        interfixes (list of str, optional):
            A list of words to be considered as interfixes between components
            of compound nouns. Defaults to ['e', 'n', 's'], being the list of
            Danish interfixes.

    Attributes:
        small_dictionary (dict): Dictionary containing the small dictionary.
        dictionary (dict): Dictionary containing the dictionary.
        interfixes (list of str): Possible interfixes in compound nouns.
    '''
    def __init__(self,
                 small_dictionary: List[str],
                 dictionary: List[str],
                 interfixes: List[str] = ['e', 'n', 's']):

        small_dict = pd.DataFrame.from_dict(dict(words=small_dictionary))
        small_dict['length'] = small_dict.words.str.len()
        small_dict = small_dict.sort_values(by='length')
        self.small_dictionary = {
            length: small_dict.query('length == @length').words.tolist()
            for length in range(1, small_dict.length.max())
        }

        dictionary = pd.DataFrame.from_dict(dict(words=dictionary))
        dictionary['length'] = dictionary.words.str.len()
        dictionary = dictionary.sort_values(by='length')
        self.dictionary = {
            length: dictionary.query('length == @length').words.tolist()
            for length in range(1, dictionary.length.max())
        }
        self.interfixes = interfixes

    def _lookup(self, string: str, all_cases: bool = False) -> bool:
        '''Check if the given string is a dictionary word.

        Args:
            string (str):
                The string to check.
            all_cases (bool, optional):
                Whether to check if the string is a dictionary word in all
                cases. Defaults to False.

        Returns:
            bool: True if the string is a dictionary word, False otherwise.
        '''
        if not all_cases:
            return string in self.small_dictionary.get(len(string), [])
        else:
            return string in self.dictionary.get(len(string), [])

    def split(self,
              _: int,
              normalized_string: NormalizedString) -> List[NormalizedString]:
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
            if self._lookup(string[:char_idx]):

                remaining = normalized_string[char_idx + 1:]
                remaining_splits = self.split(0, remaining)

                if (len(remaining_splits) == 1 and
                        not self._lookup(str(remaining_splits[0]),
                                         all_cases=True)):
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
        pretok.split(self.split)
