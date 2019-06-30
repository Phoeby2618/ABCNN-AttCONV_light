import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer,xavier_initializer,xavier_initializer_conv2d


class ABCNN(object):
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
        self.block=config.block

        self.cnn_outnum=config.cnn_outnum
        self.filter_width=config.filter_width
        self.pooling_width=config.pooling_width

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
        self.out1=[]
        self.out2=[]
        s1_in = self.s1_embed       #[bs,len,hidden]
        s2_in = self.s2_embed

        for i in range(self.block):
            with tf.variable_scope('{}_block'.format(i+1),tf.AUTO_REUSE):
                with tf.variable_scope('layer1_conv'):

                    if self.modeltype=='ABCNN1' or self.modeltype=='ABCNN3':
                        s1_map,s2_map=self.weighted_attmap(s1_in,s2_in)     #[bs,len,hidden]

                        s1_in=tf.concat([s1_in,s1_map],axis=-1)
                        s2_in = tf.concat([s2_in, s2_map], axis=-1)

                    s1_conv=self.convolution(s1_in,self.cnn_outnum,self.filter_width)    #[bs,len,cnn_outnum]
                    s2_conv=self.convolution(s2_in,self.cnn_outnum,self.filter_width)

                with tf.variable_scope('layer1_pooling'):
                    s1_in2=s1_conv
                    s2_in2=s2_conv
                    if self.modeltype=='ABCNN2' or self.modeltype=='ABCNN3':
                        s1 = tf.tile(tf.expand_dims(s1_in2, axis=2), [1, 1, self.max_len, 1])
                        s2 = tf.tile(tf.expand_dims(s2_in2, axis=1), [1, self.max_len, 1, 1])
                        eumap = self._euclidean_distance_matrix(s1, s2)         #[bs,s1,s2]
                        s1_weight=tf.reduce_sum(eumap,axis=-1,keepdims=True)    #(bs,len,1)
                        s2_weight = tf.reduce_sum(tf.transpose(eumap,perm=[0,2,1]), axis=-1, keepdims=True)
                        s1_in2=s1_in2*tf.tile(s1_weight,[1,1,self.cnn_outnum])
                        s2_in2 = s2_in2 * tf.tile(s2_weight, [1, 1, self.cnn_outnum])

                    pool_s1=self.pooling(s1_in2,self.pooling_width)             #[bs,len,out]
                    pool_s2=self.pooling(s2_in2,self.pooling_width)

                    all_pool_s1=tf.reduce_mean(pool_s1,axis=1)
                    all_pool_s2=tf.reduce_mean(pool_s2,axis=1)

                    self.out1.append(all_pool_s1)
                    self.out2.append(all_pool_s2)

                s1_in=pool_s1
                s2_in=pool_s2


    def pred(self):
        output=tf.concat([tf.concat(self.out1,axis=-1),tf.concat(self.out2,axis=-1)],axis=-1)
        self.pre=self.linear(output,self.n_class,'w_pred','b_pred',
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

    def weighted_attmap(self,s1, s2):
        '''

        :param s1: [bs,len,hidden]
        :param s2:
        :return:
        '''
        dim=s1.get_shape()[2].value
        with tf.variable_scope('attmap'):
            s1=tf.tile(tf.expand_dims(s1,axis=2),[1,1,self.max_len,1])
            s2=tf.tile(tf.expand_dims(s2,axis=1),[1,self.max_len,1,1])
            eumap=self._euclidean_distance_matrix(s1,s2)           #[bs,s1,s2]
            w0=tf.get_variable('w0',shape=[self.max_len,dim],
                               initializer=xavier_initializer(),
                               regularizer=l2_regularizer(self.l2_reg))
            w1 = tf.get_variable('w1', shape=[self.max_len, dim],
                                 initializer=xavier_initializer(),
                                 regularizer=l2_regularizer(self.l2_reg))

            s1_map=tf.tensordot(eumap,w0,axes=1)
            s2_map=tf.tensordot(tf.transpose(eumap,perm=[0,2,1]),w1,axes=1)
            return s1_map,s2_map

    @staticmethod
    def _euclidean_distance_matrix(a, b):
        '''

        :param a: [bs,seq1,seq2',hidden]
        :param b: [bs,seq1',seq2,hidden]
        :return:[bs,seq1,seq2]
        '''
        #尝试用矩阵乘法，欧式的loss会出现nan，需要给sqrt一个最小初值。
        #eumap=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(a,b)),axis=3))
        eumap=tf.reduce_sum(tf.multiply(a,b),axis=-1)
        return eumap

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
                                        weights_initializer=xavier_initializer_conv2d(),
                                        weights_regularizer=l2_regularizer(self.l2_reg)
                                        )

    def pooling(self,s,pooling_width):
        #一维平均池化只有tf>1.4以上版本有。
        '''

        :param s:[bs,len,hidden]
        :param pooling_width: pooling width
        :return:
        '''
        s=tf.expand_dims(s,1)
        pool=tf.nn.avg_pool(s,
                              [1,1,pooling_width,1],
                              [1,1,1,1],
                              padding='SAME')
        return tf.squeeze(pool,axis=1)



if __name__ == '__main__':
    import config

    config = config.parser_args()
    config.word_vocab_size = 5
    config.char_vocab_size = 6
    config.is_train = True
    model = ABCNN(config)









