import DeepLearning.Embedding.fasttext as FT
import kss
import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

text_array=[]
with open("../LanguageModel/test.txt", "r", encoding='utf-8') as f:
    while True:
        data = f.readline().strip()
        if not data: break
        sent = kss.split_sentences(data)
        text_array.extend(sent)

# GPU 할당 변경하기
GPU_NUM = 0  # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
torch.cuda.set_per_process_memory_fraction(0.8, 0)
print('# Current cuda device: ', torch.cuda.current_device())  # check

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

ko_model = FT.load_fasttext()
for s in sent:
    input_ids = tokenizer.tokenize(s)
    print(input_ids)
    for i in range(len(input_ids)):
        if input_ids[i][0] != '▁':
            res=FT.similar_word(ko_model,input_ids[i])
            input_ids[i] = res[0][0]

print(' '.join(input_ids).replace(' ','').replace('▁', ' '))



