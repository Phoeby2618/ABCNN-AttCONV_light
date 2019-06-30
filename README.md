ABCNN

不完全实现ABCNN和Attentive_light的版本

[ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](<https://www.aclweb.org/anthology/Q16-1019>)

[Attentive Convolution:Equipping CNNs with RNN-style Attention Mechanisms](<https://arxiv.org/pdf/1710.00519.pdf>)	

## Requirements

- Python>=3
- TensorFlow==1.12
- Numpy
- NLTK >=3.4

## Dataset

用于各种句子对任务，文本蕴含SNLI，答案选择WIKIQA，释义识别MSPR等.

## Model Intro

主要是两个CNN用于句子匹配任务的模型：

ABCNN模型提出了ABCNN1,ABCNN2,ABCNN33个框架，将词级别注意力矩阵以下图（连接和加权）两种方式和原句子进行组合，再用cnn进行卷积。

本库实现和原文不同点：1.词级矩阵注意力用元素点积方式，2.文中用宽卷积，本文直接SAME方式，3.输出直接softmax映射。

![ABCNN](/img/ABCNN.PNG 'ABCNN')

Attentive_light则是利用短语级（邻近词信息）加权信息在卷积过程中进行注意力，而非将注意力用在池化上。

![Att_light](/img/Att_light.png)



## Usage

1. Download dataset

2. Modify file paths in ` config.py` 

3. Training model

   ```
   python run.py
   ```

   

