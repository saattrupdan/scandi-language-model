'''Push the model the HF Hub'''

from transformers import AutoModelForPreTraining, AutoTokenizer
import sys


def main():
    '''Push the model to the HF Hub'''
    if len(sys.argv) == 2:
        model_id = sys.argv[-1]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.push_to_hub(model_id)
        model = AutoModelForPreTraining.from_pretrained(model_id)
        model.push_to_hub(model_id)


if __name__ == '__main__':
    main()
