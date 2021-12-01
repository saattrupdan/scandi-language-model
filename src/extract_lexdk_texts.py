'''Extracts the texts of a scraped LexDk corpus.

Usage:
    python extract_lexdk_texts.py <lexdk_jsonl_dump>
'''

import sys
import json
from pathlib import Path
from tqdm.auto import tqdm


def main():
    file_in = Path(sys.argv[1])
    file_out = file_in.parent / f'{file_in.stem}.txt'

    with file_out.open('w', encoding='utf-8') as f_out:
        with file_in.open('r', encoding='utf-8') as f_in:
            desc = f'Extracting texts from {file_in}'
            for line in tqdm(f_in, desc=desc):
                data = json.loads(line)
                f_out.write(data['text'])


if __name__ == '__main__':
    main()
