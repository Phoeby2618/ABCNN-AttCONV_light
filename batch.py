import numpy as np
import tensorflow.contrib.keras as kr
from dataprocess import Dataprocess
import time

class BatchGenerator:
    '''
    获取batch，
    填充字符或词长度，，
    生成词性one-hot
    '''
    def __init__(self,dataset,shuffle=False,word_pad=0,maxsen_len=0):
        '''

        :param dataset: [{k1:v1,k2:v2,k3:v3,...},{k1:v1,k2:v2,k3:v3,...},{k1:v1,k2:v2,k3:v3,...},...]
        :param shuffle:
        :param word_pad:
        :param char_pad:
        :param char_len:
        :param match_pad:
        :param pos_pad:
        :param pos_len:     #上6可根据config得到，需要的话根据具体任务而定。
        '''
        self.dataset=np.array(dataset)
        self.data_size=len(dataset)
        self.start=0
        self.padnum=word_pad
        self.maxsen_len=maxsen_len

        if shuffle:
            random_indices=np.random.permutation(self.data_size)
            self.dataset=self.dataset[random_indices]

    def get_size(self):
        return self.data_size

    def next_batch(self,batchsize):

        start=self.start
        end=start+batchsize
        if end>=self.data_size:
            random_indices = np.random.permutation(self.data_size)
            self.dataset = self.dataset[random_indices]
            self.start=0
            start = self.start
            end = batchsize

        batchdataset=self.dataset[start:end]
        batchdata=self.getbatch(batchdataset)    #填充，根据具体任务变化，需哪些输出
        self.start=end

        return batchdata

    #根据不同数据集而变化

    def getbatch(self, batchdataset):
        #batch词长度填充，字符长度填充&字符到句长度填充，匹配特征expand_dim，POS的填充和one_hot
        label=[ d['label'] for d in batchdataset]

        s1_len=[ d['sen1_len'] for d in batchdataset]
        s2_len = [ d['sen2_len'] for d in batchdataset]

        if self.maxsen_len>0:
            maxsen_len=self.maxsen_len
        else:
            maxsen_len=max(max(s1_len),max(s2_len))

        s1 = [data['sen1wordID'] for data in batchdataset]
        s2 = [data['sen2wordID'] for data in batchdataset]
        #填充句子长度
        s1 = kr.preprocessing.sequence.pad_sequences(s1, maxsen_len, padding='post', value=self.padnum)
        s2 = kr.preprocessing.sequence.pad_sequences(s2, maxsen_len, padding='post', value=self.padnum)

        s1_mask = (s1 != self.padnum).astype(np.int32)
        s2_mask = (s2 != self.padnum).astype(np.int32)


        return s1,s2,label,s1_len,s2_len,s1_mask,s2_mask

if __name__=='__main__':

    print('开始构建：',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    file_path='data/snli_1.0_dev.jsonl'
    datapro=Dataprocess(file_path)
    testdata=np.array(datapro.processdata)
    print(len(testdata))
    print(len(testdata[0]))

    batches=BatchGenerator(testdata,True)
    print('batch完毕!',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    batch_size=32
    batchnum=int(batches.data_size/batch_size)
    for i in range(batchnum):
        s1,s2,label,s1_len,s2_len,s1_mask,s2_mask=batches.next_batch(32)
        print(i,np.array(s1).shape)

    # print(s1)
    # print(label)
    # print(s1_match.shape)
    # print(s2_match.shape)
    # print(s1_pos.shape)
    # print(s2_pos.shape)








