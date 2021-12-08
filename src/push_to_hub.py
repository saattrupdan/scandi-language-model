'''Push the model the HF Hub'''

from transformers import AutoModelForPreTraining


def main():
    '''Push the model to the HF Hub'''
    model = AutoModelForPreTraining.from_pretrained('roberta-base-wiki-da')
    model.push_to_hub('roberta-base-wiki-da')


if __name__ == '__main__':
    main()
