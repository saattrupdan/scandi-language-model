'''Split a file full of line-delimited texts into sentences.

For Wikipedia data, first call the `extract_wiki_texts.sh` script to generate a
cleaned Wikipedia dump file, and call this script on the resulting file to
preprocess it.

Usage:
    python split_into_sentences.py <cleaned_input_file>
'''

import sys
from pathlib import Path
from tqdm.auto import tqdm
from blingfire import text_to_sentences


def main():
    file_in = Path(sys.argv[1])
    file_out = file_in.parent / f'{file_in.stem}-preprocessed{file_in.suffix}'

    with file_out.open('w', encoding='utf-8') as f_out:
        with file_in.open('r', encoding='utf-8') as f_in:
            desc = f'Sentence-splitting {file_in} to {file_out}...'
            for line in tqdm(f_in, desc=desc):
                sentences = [sent
                             for sent in text_to_sentences(line).split('\n')
                             if sent != '' and ' ' in sent]
                f_out.write('\n'.join(sentences) + '\n')



if __name__ == '__main__':
    main()
