# need kobart package
# pip install git+https://github.com/SKT-AI/KoBART#egg=kobart  torch 1.71...
# BartModel인데 BartForConditionalGeneration로 바뀜. 이슈 해결한 class라고 함
from transformers import BartForConditionalGeneration,PreTrainedTokenizerFast
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
kobart_tokenizer = get_kobart_tokenizer()
model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
#kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained(get_pytorch_kobart_model())

inputs = kobart_tokenizer.encode('오늘은 김치찌개 만드는 법에 대해 ', return_tensors='pt')
gen_ids=model.generate(input_ids=inputs,     bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    length_penalty=2.0,
    max_length=142,
    min_length=56,
    num_beams=4,)
generated = kobart_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
print(generated)