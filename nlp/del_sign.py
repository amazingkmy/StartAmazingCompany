import re

def clean_text(inputString):
  text_rmv = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', inputString)
  text_rmv = ' '.join(text_rmv.split())
  return text_rmv

if __name__ == '__main__':
    datalist=[]
    splitdata=[]
    hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    with open("./res.res", "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip(): continue
            clean_line = clean_text(line)
            c_line = re.fullmatch('[a-zA-Z0-9ㄱ-ㅣ가-힣 ]+', clean_line)
            if c_line is not None:
                datalist.append(clean_line)
                clean_sline = clean_line.split(' ')
                for c in clean_sline:
                    if not c: continue
                    # 영어
                    if c.isalpha():
                        splitdata.append(c); continue;
                    hg = re.sub('[^ \u3131-\u3163\uac00-\ud7a3]+', '', c)
                    # 한글만
                    if c == hg: splitdata.append(c)
                    # 한글, 영어, 숫자 로 구성된 것 제외 삭제
                    ukword=re.fullmatch('[a-zA-Z0-9ㄱ-ㅣ가-힣 ]+', c)
                    if ukword is not None:
                        splitdata.append(c)
    set_sdata = set(splitdata)
    sdata_array= list(set_sdata)
    set_line = set(datalist)
    list_array= list(set_line)

    with open("./clean_res.res", "w", encoding='utf-8') as f:
        for d in list_array:
            f.write(''.join(d.split(' '))+'/NNG')
            f.write('\n')
    with open("./word_res.res", "w", encoding='utf-8') as f:
        for s in sdata_array:
            f.write(s)
            f.write('\n')