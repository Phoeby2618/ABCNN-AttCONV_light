
import json
import os
import pickle
import re
import collections
import nltk
import numpy as np

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": -1
}

class Dataprocess:
    def __init__(self,data_file,word_dict_file,use_pos_fea=True,char_len=16):
        self.char_len=char_len
        #获取数据,这里可用词组一起获取。
        dataset=self.get_data(data_file)
        self.get_dict([dataset],word_dict_file)
        self.get_id(dataset)

        if use_pos_fea:
            self.exact_match(dataset)
            POS_PADDING = ['<PAD>','<UNK>']
            POS_Tagging = POS_PADDING+[ 'WP$', 'RBS', 'SYM', 'WRB', 'IN', 'VB', 'POS', 'TO', ':', '-RRB-', '$', 'MD', 'JJ', '#',
                           'CD', '``', 'JJR', 'NNP', "''", 'LS', 'VBP', 'VBD', 'FW', 'RBR', 'JJS', 'DT', 'VBG', 'RP', 'NNS',
                           'RB', 'PDT', 'PRP$', '.', 'XX', 'NNPS', 'UH', 'EX', 'NN', 'WDT', 'VBN', 'VBZ', 'CC', ',',
                           '-LRB-', 'PRP', 'WP']
            self.POS_dict = {pos: i for i, pos in enumerate(POS_Tagging)}
            self.pos_len = len(POS_Tagging)
            print('pos_len:',self.pos_len )

            self.pos_feature(dataset)


    def get_data(self,file):
        '''
                根据不同的数据文件处理的方式可能会有变化，但是得到的最后token形式不变。
        :param file:
        :return: tokendata[{pairID=,sentence1,sentence2,label},{}, ...]
        '''
        # 直接用原句子中解析的分词，也可自己分词
        def tokenize(string):
            string = re.sub(r'\(|\)', '', string)
            return string.split()

        #用nltk分句子词比上述结果慢
        def tokenize1(string):
            wordslist = nltk.word_tokenize(string)
            return wordslist

        tokendata=[]
        with open(file, 'r', encoding='utf-8') as fr:
            for i, lines in enumerate(fr):
                l = lines.strip().split('\t')
                data={}

                data['pairID']=i
                data['sentence1']=tokenize1(l[0].strip())
                data['sentence2']=tokenize1(l[1].strip())
                data['label']=l[2]

                tokendata.append(data)

        return tokendata

    def get_dict(self,datasets,word_dict_file,filter_threshold=0):
        '''
        #这里应该需要很多个文件来找字典
        :param datasets: 已分好词的数组
        :param filter_threshold: 过滤低频字符门槛
        :return: 词计数，词典，词to索引，索引to词；
                字符计数，字典，字符to索引，索引to字符
        '''
        if os.path.exists(word_dict_file):
            print('word dict file exisit!')
            with open(word_dict_file,'rb') as f:
                dictdata=pickle.load(f)
            self.word_vocab,self.char_vocab,self.word_to_id,self.id_to_word,self.char_to_id,self.id_to_char=dictdata

        else:
            self.word_count=collections.Counter()
            self.char_count=collections.Counter()

            for dataset in datasets:
                for i,d in enumerate(dataset):
                        self.word_count.update(d['sentence1'])
                        self.word_count.update(d['sentence2'])

                        for word in d['sentence1']:
                            self.char_count.update(word)
                        for word in d['sentence2']:
                            self.char_count.update(word)
            if filter_threshold:
                word_vocab = [word for word in set(self.word_count.keys()) if self.word_count[word]>filter_threshold]
                char_vocab = [ch for ch in set(self.char_count.keys()) if self.char_count[ch]>filter_threshold]
            else:
                word_vocab=list(set(self.word_count.keys()))
                char_vocab = list(set(self.char_count.keys()))

            extra=['<PAD>','<UNK>']
            self.word_vocab=extra[:]+word_vocab
            self.char_vocab=extra[:]+char_vocab

            self.word_to_id=dict(zip(self.word_vocab,range(len(self.word_vocab))))
            self.id_to_word={v:k for k,v in self.word_to_id.items()}

            self.char_to_id=dict(zip(self.char_vocab,range(len(self.char_vocab))))
            self.id_to_char={v:k for k,v in self.char_to_id.items()}

            #看是否需要存，存的话需要在字典中说明过滤门槛数，避免混淆
            with open(word_dict_file,'wb') as fdw:
                pickle.dump((self.word_vocab,self.char_vocab,self.word_to_id,
                             self.id_to_word,self.char_to_id,self.id_to_char),fdw)

    def get_id(self,dataset):
        '''

        :param dataset: 单个数据处理id
        :return: 如果只有一个数据处理，则可self，否则需要返回，可写一个公共的得到所有处理的函数
        '''
        processdata=[]
        for data in dataset:
            isprocess={}
            isprocess['pairID']=data['pairID']
            isprocess['sen1wordID']=[self.word_to_id[word] if self.word_to_id.get(word) else self.word_to_id['<UNK>']
                                     for word in data['sentence1']]
            isprocess['sen2wordID'] = [self.word_to_id[word] if self.word_to_id.get(word) else self.word_to_id['<UNK>']
                                        for word in data['sentence2']]

            isprocess['sen1_len']=len(data['sentence1'])
            isprocess['sen2_len'] = len(data['sentence2'])

            isprocess['sen1charID']=[[self.char_to_id[word[i]] if i<len(word) else self.char_to_id['<PAD>']
                                      for i in range(self.char_len)] for word in data['sentence1']]
            isprocess['sen2charID']=[[self.char_to_id[word[i]] if i<len(word) else self.char_to_id['<PAD>']
                                      for i in range(self.char_len)] for word in data['sentence2']]

            isprocess['label']=data['label']
            processdata.append(isprocess)

        self.processdata=processdata

    def get_processdata(self):
        return self.processdata

    def get_vocab_size(self):
        return len(self.word_vocab)

    def exact_match(self,dataset):
        '''
        1.这里先用全匹配，不用词根方式，全匹配借用字典可降低复杂度为o（n）
        若用词根也要o（n）则需先提取每个词的词根

        2.形式:最开始想的是直接一个one-hot连接在最后分类，
            根据本模型代码s1，s2分别匹配one-hot特征，哪个对上哪个为1,后面为防止一个batch长度不等还需要padding0
        :param dataset:
        :return:
        '''
        for j,data in enumerate(dataset):
            s1=data['sentence1']
            s2 = data['sentence2']
            max_len=max(len(s1),len(s2))
            s1_match=[0]*max_len
            s2_match = [0] * max_len

            s1_dict=dict(zip(s1,range(len(s1))))
            s2_dict = dict(zip(s2, range(len(s2))))
            i=0
            for word1,word2 in zip(s1,s2):
                if word1 in s2_dict:
                    s1_match[i]=1
                if word2 in s1_dict:
                    s2_match[i]=1
                i+=1

            if self.processdata[j]['pairID']==data['pairID']:
                self.processdata[j]['s1_exact_match']=s1_match
                self.processdata[j]['s2_exact_match'] = s2_match
            else:
                print( 'extact_match 顺序对不上！')
        print(self.processdata[0])

    def one_hot_pos_feature(self,sequence, seq_len,left_padding_and_cropping_pairs=(0,0), column_size=None):
        left_padding, left_cropping = left_padding_and_cropping_pairs
        mtrx = np.zeros((seq_len, column_size))
        for row, col in sequence:
            if row + left_padding - left_cropping < seq_len and row + left_padding - left_cropping >= 0 and col < column_size:
                mtrx[row + left_padding - left_cropping, col] = 1
        return np.array(mtrx)

    def parsing_parse(self, parse):
        base_parse = [s.rstrip(" ").rstrip(")") for s in parse.split("(") if ")" in s]
        pos = [pair.split(" ")[0] for pair in base_parse]
        return pos

    def pos_feature(self,dataset):
        '''

        :param dataset:
        :return:  [[(0,pos1),(1,pos2),(2,pos3), ...],[]]
        '''

        for i ,data in enumerate(dataset):
            pos1 = self.parsing_parse(data['sentence1_parse'])
            pos2 = self.parsing_parse(data['sentence2_parse'])
            pos_indices1= [(idx, self.POS_dict.get(tag, 1)) for idx, tag in enumerate(pos1)]
            pos_indices2 = [(idx, self.POS_dict.get(tag, 1)) for idx, tag in enumerate(pos2)]
            if self.processdata[i]['pairID'] == data['pairID']:
                s1_len=self.processdata[i]['sen1_len']
                s2_len = self.processdata[i]['sen2_len']
                s1_pos_vec=self.one_hot_pos_feature(pos_indices1,s1_len,column_size=self.pos_len)
                s2_pos_vec=self.one_hot_pos_feature(pos_indices2, s2_len, column_size=self.pos_len)

                self.processdata[i]['parse_s1'] = s1_pos_vec
                self.processdata[i]['parse_s2'] = s2_pos_vec
            else:
                print('解析顺序对不上！')
        # print(self.processdata[0])
        # print(self.processdata[0]['parse_s1'].shape)
        # print(self.processdata[0]['parse_s2'].shape)

    def get_char_dict(self):
        return self.char_vocab

    def get_word_dict(self):
        return self.word_vocab



if __name__=='__main__':
    file_path='data/snli_1.0_dev.jsonl'
    datapro=Dataprocess(file_path,'word_dict_file.pkl')
