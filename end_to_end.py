# -*- coding: utf-8 -*-
import sys, re, jieba
import argparse, json, time
import numpy as np
import tensorflow as tf

# 以下导入的都是fsauor2018文件夹下的本地文件
sys.path.append("./models/fsauor2018")
from dataset import DataSet
from model import Model
from utils import *
from thrid_utils import read_vocab


def flags():
    return
    
fsauor_path = "./models/fsauor2018"

flags.data_files=fsauor_path+"/scripts/data/testa.json"
flags.label_file=fsauor_path+"/scripts/data/labels.txt"
flags.vocab_file=fsauor_path+"/scripts/data/vocab.txt"
flags.out_file=fsauor_path+"/scripts/data/out.testa.json"
flags.prob=False
flags.batch_size=300
flags.feature_num=20
flags.checkpoint_dir=fsauor_path+"/scripts/data/elmo_ema_0120"
flags.embed_file=fsauor_path+"/scripts/data/embedding.txt"
flags.reverse=False
flags.split_word=True
flags.max_len=1200

def replace_dish(content):
    return re.sub("【.{5,20}】","<dish>",content)

def normalize_num(words):
    '''Normalize numbers
    for example: 123 -> 100,  3934 -> 3000
    '''
    tokens = []
    for w in words:
        try:
            ww = w
            num = int(float(ww))
            if len(ww) < 2:
                tokens.append(ww)
            else:
                num = int(ww[0]) * (10**(len(str(num))-1))
                tokens.append(str(num))
        except:
            tokens.append(w)
    return tokens

def tokenize(content):
    content = content.replace("\u0006",'').replace("\u0005",'').replace("\u0007",'')
    tokens = []
    content = content.lower()
    # 去除重复字符
    content = re.sub('~+','~',content)
    content = re.sub('～+','～',content)
    content = re.sub('(\n)+','\n',content)
    for para in content.split('\n'):
        para_tokens = []
        words = list(jieba.cut(para))
        words = normalize_num(words)
        para_tokens.extend(words)
        para_tokens.append('<para>')
        tokens.append(' '.join(para_tokens))
    content = " ".join(tokens)
    content = re.sub('\s+',' ',content)
    content = re.sub('(<para> )+','<para> ',content)
    content = re.sub('(- )+','- ',content)    
    content = re.sub('(= )+','= ',content)
    content = re.sub('(\. )+','. ',content).strip()
    content = replace_dish(content)
    if content.endswith("<para>"):
        content = content[:-7]
    return content

def _preprocess(self):
    print_out("# Start to preprocessing data...")
    for fname in self.data_files:
        print_out("# load data from %s ..." % fname)
        for line in open(fname, 'r', encoding="utf-8"):
            item = json.loads(line.strip())
            content = item['content']
            content = _tokenize(content, self.w2i, self.max_len, self.reverse, self.split_word)
            item_labels = []
            for label_name in self.label_names:
                labels = [item[label_name]]
                labels = self.get_label(labels,self.tag_l2i)
                item_labels.append(labels)
            self._raw_data.append(DataItem(content=content,labels=np.asarray(item_labels),length=len(content),id=int(item['id'])))
            self.items.append(item)

    self.num_batches = len(self._raw_data) // self.batch_size
    self.data_size = len(self._raw_data)
    print_out("# Got %d data items with %d batches" % (self.data_size, self.num_batches))


vocab, w2i = read_vocab(flags.vocab_file)

UNK_ID = 0
SOS_ID = 1
EOS_ID = 2

def _tokenize(content, w2i, max_tokens=1200, reverse=False, split=True):
    def get_tokens(content):
        tokens = content.strip().split()
        ids = []
        for t in tokens:
            if t in w2i:
                ids.append(w2i[t])
            else:
                for c in t:
                    ids.append(w2i.get(c,UNK_ID))
        return ids
    if split:
        ids = get_tokens(content)
    else:
        ids = [w2i.get(t,UNK_ID) for t in content.strip().split()]
    if reverse:
        ids = list(reversed(ids))
    tokens = [SOS_ID] + ids[:max_tokens] + [EOS_ID]
    return tokens

def initialize_model():
    # dataset = DataSet(flags.data_files, flags.vocab_file, flags.label_file, flags.batch_size, reverse=flags.reverse, split_word=flags.split_word, max_len=flags.max_len)
    hparams = load_hparams(flags.checkpoint_dir,{"mode":'inference','checkpoint_dir':flags.checkpoint_dir+"/best_eval",'embed_file':flags.embed_file, 'vocab_file':flags.vocab_file})
    
    sess = tf.Session(config = get_config_proto(log_device_placement=False))

    model = Model(hparams)
    model.build()

    try:
        model.restore_model(sess)  #restore best solution
    except Exception as e:
        print("unable to restore model with exception",e)
        exit(1)
    else:
        print("===========================")
        print("========模型加载完毕=======")
        print("===========================")
    return sess, model

    # scalars = model.scalars.eval(session=sess)
    # print("Scalars:", scalars)
    # weight = model.weight.eval(session=sess)
    # print("Weight:",weight)
    # cnt = 0

i2l = read_vocab(flags.label_file)
tag_l2i = {"1":0,"0":1,"-1":2,"-2":3}
tag_i2l = {v:k for k,v in tag_l2i.items()}
from demo import extract_text
def inference(input_text, sess, model, return_prob=False):
    source=tokenize(input_text)
    source=[_tokenize(source, w2i)]
    # print(source)
    lengths=[len(source[0])]
    predict, logits = model.inference_clf_one_batch(sess, source, lengths)
    
    output={}
    for i,(p,l) in enumerate(zip(predict,logits)):
        for j in range(flags.feature_num):
            label_name = i2l[0][j]
            if return_prob:
                tag =  [float(v) for v in l[j]]
            else:
                tag = tag_i2l[np.argmax(p[j])]
            print(label_name, tag)
            output[label_name]=tag
            # dataset.items[cnt + i][label_name] = tag
    # cnt += len(lengths)
    # print_out("\r# process {0:.2%}".format(cnt/dataset.data_size),new_line=False)    

    return output


'''
模型需要输入：source，lengths
其中source为_tokenize函数转换后的index_encoded文本
length为len(content)

_tokenize将raw文本序列（已分词）转换为index序列
'''

if __name__ == "__main__":
    sess, model2 = initialize_model()
    # source=tokenize("我今天吃了一个包子")
    # source=_tokenize(source, w2i)
    # print(source)
    while True:
        print("\n")
        text = input("==========================\n输入文本（输入exit退出）：\n==========================\n")
        if text == "exit": break
        inference(text, sess=sess, model=model2)

