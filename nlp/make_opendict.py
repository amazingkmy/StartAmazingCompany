import xml.etree.ElementTree as ET
import os
import argparse
from os import listdir
from os.path import isfile, join
import pickle


poslist=[]
def set_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='opendict_path')
    parser.add_argument('--output', required=False, default='./nlp_dict', help='savedict_path')
    return parser

def read_xml(filelist):
    xmllist=[]
    for filename in filelist:
        if filename.find('xml') == -1: continue
        print(filename)
        tree = ET.parse(filename)
        root = tree.getroot()
        xmllist.append(root)
    return xmllist

def get_opendict_xml(xmllist):
    opendict={}
    i = 0
    for xmldata in xmllist:
        for item in xmldata.findall('item'):
            try:
                targetcode = item.find('target_code').text
                word = item.find('wordInfo/word').text
                pos = item.find('senseInfo/pos').text
                if pos not in poslist: poslist.append(pos)
                relation = item.findall('senseInfo/relation_info')
                sameword=[]
                for r in relation:
                    sameword.append(r.find('link_target_code').text)
                opendict[targetcode]=[word, pos, sameword]
                if i % 10000 == 0: print(i)
                i+=1
            except: continue
    print(i)
    return opendict

def save_dict(filename, opendict):
    with open(filename, "wb") as f:
        pickle.dump(opendict, f)


if __name__ == '__main__':
    parser = set_parse()
    args = parser.parse_args()
    inputfiles = [args.input+'/'+f for f in listdir(args.input) if isfile(join(args.input, f))]

    # XML Read
    xmllist = read_xml(inputfiles)
    # Get Dict Information
    opendict = get_opendict_xml(xmllist)

    # Save Dict File
    save_dict(args.output, opendict)

    with open("./poslist.list","wt") as f:
        for p in poslist:
            f.write(p)
            f.write('\n')
