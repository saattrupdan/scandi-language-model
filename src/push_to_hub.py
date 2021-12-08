'''Push the model the HF Hub'''

from transformers import AutoModelForPreTraining, PreTrainedTokenizerFast
import sys


def main():
    '''Push the model to the HF Hub'''
    if len(sys.argv) == 3:
        model_id = sys.argv[1]
        tokenizer_id = sys.argv[2]

        # Load pretrained tokenizer and push it to the hub
        tokeniser = PreTrainedTokenizerFast(tokenizer_file=tokenizer_id,
                                            bos_token='<s>',
                                            cls_token='<s>',
                                            eos_token='</s>',
                                            sep_token='</s>',
                                            unk_token='<unk>',
                                            mask_token='<mask>',
                                            pad_token='<pad>')
        tokeniser.push_to_hub(model_id)

        # Load pretrained model and push it to the hub
        model = AutoModelForPreTraining.from_pretrained(model_id)
        model.push_to_hub(model_id)


if __name__ == '__main__':
    main()
