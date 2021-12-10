'''Combine all the datasets as a HuggingFace dataset'''

from datasets import Dataset
from pathlib import Path
import pandas as pd


def main():
    '''Main function'''

    # Set up the paths
    csv_path = Path('data') / 'retskrivningsordbog.csv'
    txt_path = Path('data') / 'retskrivningsordbog.txt'

    # Extract and store the words
    (pd.read_csv(csv_path, sep=';', names=['baseform', 'word', 'type'])
       .word
       .str.replace('[0-9]+[.] ', '', regex=True)
       .str.lower()
       .drop_duplicates()
       .sort_values(key=lambda x: x.str.len())
       .to_csv(txt_path, index=False, header=False))

    # Create the dataset
    dataset = Dataset.from_text(str(txt_path))

    # Save the dataset
    dataset.save_to_disk('data/dict_dataset')


if __name__ == '__main__':
    main()
