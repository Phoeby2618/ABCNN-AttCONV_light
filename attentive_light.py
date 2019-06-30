import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer,xavier_initializer,xavier_initializer_conv2d
from utils import mask_softmax


class AttentiveLight(object):
    def __init__(self, config):
        '''

        :param config:
        '''
        self.keep_prob = config.drop_keep_prob
        self.is_train = tf.cast(config.is_train, tf.bool)
        self.n_class = config.n_class
        self.optimizer = config.optimizer
        self.clip_value = config.clip_value
        self.l2_reg = config.l2_reg

        self.word_embeddings = config.word_embedding
        self.word_embed_dim = config.word_embed_size
        self.word_vocab_size = config.word_vocab_size

        self.max_len = config.max_seq_len
        self.modeltype=config.modeltype

        self.attentive_out=config.attentive_out
        self.attentive_filter=config.attentive_filter
        self.attentive_attfilter = config.attentive_attfilter

        self.build_graph()

    def build_graph(self):
        self._placeholder_init()
        self._embedding()

        self.model()
        self.pred()

        self.accu()
        self.loss_op()
        self.train_op()

    def _placeholder_init(self):
        self.s1 = tf.placeholder(tf.int32, [None, self.max_len], name='s1')
        self.s2 = tf.placeholder(tf.int32, [None, self.max_len], name='s2')

        self.s1_mask = tf.placeholder(tf.int32, [None, self.max_len], name='mask_s1')
        self.s2_mask = tf.placeholder(tf.int32, [None, self.max_len], name='mask_s2')

        self.s1_len = tf.placeholder(tf.int32, [None], name='s1_len')
        self.s2_len = tf.placeholder(tf.int32, [None], name='s2_len')

        self.label = tf.placeholder(tf.int32, [None], name='label')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def _embedding(self):

        with tf.variable_scope('word_embedding'):
            if self.word_embeddings is not None:
                print('use embedding!')
                self.word_embeddings = tf.convert_to_tensor(self.word_embeddings, dtype=tf.float32,
                                                            name='word_embedding')
            else:
                print('random word embedding!')
                self.word_embeddings = tf.get_variable('word_embedding',
                                                       shape=[self.word_vocab_size, self.word_embed_dim],
                                                       dtype=tf.float32)
            print(self.word_embeddings)
            self.s1_embed = tf.nn.embedding_lookup(self.word_embeddings, self.s1)
            self.s2_embed = tf.nn.embedding_lookup(self.word_embeddings, self.s2)

    def model(self):
        with tf.variable_scope('attentive_light'):
            att_s2,_=self.matrix_map(self.s1_embed,self.s2_embed,self.s1_mask,self.s2_mask,
                                          self.max_len,self.max_len)
            s1_ws2=tf.matmul(att_s2,self.s2_embed)
            #s1 convd filter3,s2 convd filter1
            s1_conv=self.convolution(self.s1_embed,self.attentive_out,filter_width=self.attentive_filter)
            s1_ws2_conv = self.convolution(s1_ws2, self.attentive_out, filter_width=self.attentive_attfilter)

            b = tf.get_variable('b', shape=[self.attentive_out], dtype=tf.float32, initializer=tf.constant_initializer())
            att_out = tf.nn.tanh(s1_conv + s1_ws2_conv + b)

            att_out_max=tf.reduce_max(att_out,axis=1)
            self.output=att_out_max

    def pred(self):
        self.pre=self.linear(self.output,self.n_class,'w_pred','b_pred',
                             activation=tf.nn.tanh,regularizar=l2_regularizer(self.l2_reg))

    def accu(self):
        print('logits:', self.pre)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.arg_max(self.pre, dimension=1), dtype=tf.int32),
                                                   tf.cast(self.label, tf.int32)), tf.float32))

    def loss_op(self):
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.pre))
        if self.l2_reg:
            l2loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables()
                               if tensor.name.endswith("weights:0") or tensor.name.endswith('kernel:0')]) \
                     * tf.constant(self.l2_reg,
                                   dtype='float', shape=[], name='l2_regularization_ratio')
            tf.summary.scalar('l2loss', l2loss)
            self.loss += l2loss

    def train_op(self):
        with tf.name_scope('training'):
            if self.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                print('use sgd!')
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            tvars = tf.trainable_variables()
            for var in tvars:
                print(var.name, var.shape)
            grads = tf.gradients(self.loss, tvars)
            if self.clip_value > 0:
                grads, _ = tf.clip_by_global_norm(grads, self.clip_value)
            self.optim = optimizer.apply_gradients(
                zip(grads, tvars),
            )


    def linear(self,input, outsize, w_name, b_name=None, activation=None, regularizar=None):
        input_size = input.shape[-1]
        w = tf.get_variable(w_name, [input_size, outsize], regularizer=regularizar)
        out = tf.tensordot(input, w,axes=1)
        if b_name is not None:
            b = tf.get_variable(b_name, shape=[outsize])
            out = out + b
        if activation is not None:
            out = activation(out)
        return out


    def matrix_map(self,a, b,a_mask,b_mask,a_len,b_len,func='dot'):
        '''

        :param a: [bs,a_seq,hidden]
        :param b: [bs,b_seq,hidden]
        :param a_mask: [bs,a_seq]
        :param b_mask: [bs,b_seq]
        :param a_len: max_a_len of a sequence
        :param b_len:max_b_len b sequence
        :param func: type of match
        :return:
        '''
        if func=='dot':
            att_map=tf.matmul(a,tf.transpose(b,[0,2,1]))
        else :
            h=a.get_shape().as_list()[-1]
            # 'bilinear'
            w=tf.get_variable('bilinear_w',shape=[h,h],initializer=xavier_initializer(),
                              regularizer=l2_regularizer(self.l2_reg))
            att_map=tf.matmul(tf.tensordot(a,w,axes=1),tf.transpose(b,[0,2,1]))

        a_mask=tf.tile(tf.expand_dims(a_mask,1),[1,b_len,1])
        b_mask=tf.tile(tf.expand_dims(b_mask,1),[1,a_len,1])

        #att_map=[bs,a_seq,b_seq]
        att_soft_b=mask_softmax(att_map,b_mask)                                 #[bs,a_seq,b_soft_att]
        att_soft_a=mask_softmax(tf.transpose(att_map,[0,2,1]),a_mask)           #[bs,b_seq,a_soft_att]

        return att_soft_b,att_soft_a




    def convolution(self, s,output_num,filter_width):
        '''

        :param s:[bs,seq,hidden]
        :return:
        '''
        return tf.contrib.layers.conv1d(s,
                                        output_num,
                                        [filter_width],
                                        padding='SAME',
                                        data_format='NWC',
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=xavier_initializer(),
                                        weights_regularizer=l2_regularizer(self.l2_reg)
                                        )



if __name__ == '__main__':
    import config

    config = config.parser_args()
    config.word_vocab_size = 5
    config.char_vocab_size = 6
    config.is_train = True
    model = AttentiveLight(config)









