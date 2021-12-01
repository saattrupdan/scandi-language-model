'''Combine all the datasets as a HuggingFace dataset'''

from datasets import Dataset, concatenate_datasets


def main():
    '''Main function'''

    # Set up paths to the datasets
    da_lexdk_path = 'data/da-lexdk.txt'
    da_wiki_path = 'data/da-wikipedia-raw-preprocessed.txt'
    sv_wiki_path = 'data/sv-wikipedia-raw-preprocessed.txt'
    no_wiki_path = 'data/no-wikipedia-raw-preprocessed.txt'
    is_wiki_path = 'data/is-wikipedia-raw-preprocessed.txt'
    fo_wiki_path = 'data/fo-wikipedia-raw-preprocessed.txt'

    # Load the datasets as Dataset objects
    da_dataset = concatenate_datasets([Dataset.from_text(da_lexdk_path),
                                       Dataset.from_text(da_wiki_path)])
    sv_dataset = Dataset.from_text(sv_wiki_path)
    no_dataset = Dataset.from_text(no_wiki_path)
    is_dataset = Dataset.from_text(is_wiki_path)
    fo_dataset = Dataset.from_text(fo_wiki_path)

    # Concatenate and shuffle the datasets
    all_datasets = [da_dataset, sv_dataset, no_dataset, is_dataset, fo_dataset]
    dataset = concatenate_datasets(all_datasets).shuffle()

    # Save the dataset
    dataset.save_to_disk('data/dataset')


if __name__ == '__main__':
    main()
