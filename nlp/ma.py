from konlpy.tag import Kkma
import kss
import re

class nlp:
    def __init__(self):
        self.kkma=Kkma()
        self.hg_pattern=re.compile("[ㄱ-ㅎㅏ-ㅣ]+")
    def change_hg(self, doc):
        return re.sub(self.hg_pattern, "", doc).strip()
    def run_ma_sent(self, sentence):
        return self.kkma.pos(sentence)
    def run_ma_array(self, array):
        res=[]
        for d in array:
            res.append(self.kkma.pos(d))
        return res
    def split_sent(self, doc):
        return kss.split_sentences(doc)
    def get_noun(self, sentence):
        return self.kkma.nouns(sentence)

if __name__ == "__main__":
    nlp=nlp()
    norm_doc=nlp.change_hg("ㅠㅠ 보고싶어... 지금 어디야?")
    sents=nlp.split_sent(norm_doc)
    result=nlp.run_ma_array(sents)
    print(result)