'''Combine all the datasets as a HuggingFace dataset'''

from datasets import Dataset, concatenate_datasets, interleave_datasets
import datasets


datasets.set_caching_enabled(False)


def main():
    '''Main function'''

    # Set up paths to the datasets
    da_lexdk_path = 'data/da-lexdk-preprocessed.txt'
    da_wiki_path = 'data/da-wikipedia-raw-preprocessed.txt'
    sv_wiki_path = 'data/sv-wikipedia-raw-preprocessed.txt'
    nb_wiki_path = 'data/nb-wikipedia-raw-preprocessed.txt'
    nn_wiki_path = 'data/nn-wikipedia-raw-preprocessed.txt'
    is_wiki_path = 'data/is-wikipedia-raw-preprocessed.txt'
    fo_wiki_path = 'data/fo-wikipedia-raw-preprocessed.txt'

    # Load the datasets as Dataset objects
    da_dataset = concatenate_datasets([Dataset.from_text(da_lexdk_path),
                                       Dataset.from_text(da_wiki_path)])
    sv_dataset = Dataset.from_text(sv_wiki_path)
    no_dataset = concatenate_datasets([Dataset.from_text(nb_wiki_path),
                                       Dataset.from_text(nn_wiki_path)])
    is_dataset = Dataset.from_text(is_wiki_path)
    fo_dataset = Dataset.from_text(fo_wiki_path)

    # Concatenate and shuffle the datasets
    # all_datasets = [da_dataset, sv_dataset, no_dataset]#, is_dataset, fo_dataset]
    # dataset = interleave_datasets(all_datasets).shuffle()

    # Save the dataset
    da_dataset.save_to_disk('data/da_dataset')


if __name__ == '__main__':
    main()
