import json

filename = "./namuwiki_20210301.json"
json_title=[]
with open(filename, "r", encoding="utf-8") as f:
    json_data = json.load(f)
    json_title=[ x.get("title") for x in json_data]

resname='./res.res'
with open(resname,"w", encoding="utf-8") as f:
    for t in json_title:
        f.write(t)
        f.write('\n')