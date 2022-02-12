import torch
import kss
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == '__main__':
    text_array=[]
    gen_text = []

    with open("./test.txt", "r", encoding='utf-8') as f:
        while True:
            data = f.readline().strip()
            if not data: break
            sent=kss.split_sentences(data)
            text_array.extend(sent)
    # GPT2
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    for text in text_array:
        input_ids = tokenizer.encode(' '.join(text.split(' ')[:4]))
        gen_ids = model.generate(torch.tensor([input_ids]),
                                   max_length=128,
                                   repetition_penalty=2.0,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id,
                                   bos_token_id=tokenizer.bos_token_id,
                                   use_cache=True, num_beams=3)
        generated = tokenizer.decode(gen_ids[0,:].tolist())
        gen_text.append(generated.split('.')[0]+'.')
    # GPT3 - KAKAO - PyTorch 1.9 이상 ( Python 3.7에서 돌려본 결과 버전이 낮아서 실행하기 어려움 )
    """
    tokenizer = AutoTokenizer.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )
    model = AutoModelForCausalLM.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype='auto', low_cpu_mem_usage=True
    ).to(device='cpu', non_blocking=True)
    _ = model.eval()
    for text in text_array:
        with torch.no_grad():
            tokens = tokenizer.encode(text, return_tensors='pt').to(device='cuda', non_blocking=True)
            gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=64)
            generated = tokenizer.batch_decode(gen_tokens)[0]
            print(generated)
            gen_text.append(generated)
    """
    with open("./result.txt", "w", encoding='utf-8') as f:
        for t in gen_text:
            f.write(t)
            f.write('\n')