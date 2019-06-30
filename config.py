import argparse

def parser_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--predict', action='store_true',
                        help='predict the the model on test set')

    path_setting=parser.add_argument_group('path setting')
    path_setting.add_argument("--data_dir", type=str, default="../data/prosciTail/")
    path_setting.add_argument("--train_file", type=str, default="scitail_train.txt")
    path_setting.add_argument("--test_file", type=str, default="scitail_test.txt")
    path_setting.add_argument("--dev_file", type=str, default="scitail_dev.txt")

    path_setting.add_argument("--word_dict_file", type=str,
                              default="../data/word_dict/sciTail_word_dict.pkl")
    path_setting.add_argument("--embedding_dir", type=str,
                              default="../data/embedding/")
    path_setting.add_argument("--embedding_file", type=str,
                              default="servernew_sciTail_embedding.pkl")
    path_setting.add_argument("--save_path", type=str, help="save_path",
                              default="result/save_model/sg3FEmatch2/")
    path_setting.add_argument("--restore_path", type=str, help="a model to restore",
                              default="result/retore_model/")
    path_setting.add_argument("--summary", type=str, help="paint train_summary",
                              default="result/summary/s1/")


    model_setting=parser.add_argument_group('model setting')
    model_setting.add_argument('--word_embedding', type=None,
                               help='word embedding of training')
    model_setting.add_argument('--word_embed_size', type=int, default=300,
                               help='size of word embeddings')
    model_setting.add_argument('--hidden_size',type=int,default=300,
                               help='size of model hidden units')
    model_setting.add_argument('--word_vocab_size',type=int,
                               help='size of dict')
    model_setting.add_argument('--n_class', type=int, default=2,
                               help='number of classify')

    model_setting.add_argument('--modeltype', type=str, default='ABCNN3',
                               help='abcnn type (ABCNN1,ABCNN2,ABCNN3)')
    model_setting.add_argument('--block', type=int, default=1,
                               help='abcnn type (ABCNN1,ABCNN2,ABCNN3)')
    model_setting.add_argument('--max_seq_len', type=int, default=32,
                               help='whether share in inference lstm weights')
    model_setting.add_argument('--cnn_outnum', type=int, default=128,
                               help='whether share in inference lstm weights')
    model_setting.add_argument('--filter_width', type=int, default=3,
                               help='whether share in inference lstm weights')
    model_setting.add_argument('--pooling_width', type=int, default=3,
                               help='whether share in inference lstm weights')

    model_setting.add_argument('--attentive_out',type=int,default=256,
                               help='attentive light model convolution output')
    model_setting.add_argument('--attentive_filter', type=int, default=3,
                               help='attentive light model convolution filter width')
    model_setting.add_argument('--attentive_attfilter', type=int, default=1,
                               help='attentive light model attention convolution filter width')


    train_setting=parser.add_argument_group('train_setting')
    train_setting.add_argument('--is_train', type=bool, default=False,
                               help='whether is training or evaluate')
    train_setting.add_argument('--batch_size',type=int,default=32,
                               help='batch_size of training')
    train_setting.add_argument('--drop_keep_prob', type=float, default=0.8,
                               help='dropout keep prob')
    train_setting.add_argument('--l2_reg', type=float, default=0.0001,
                               help='l2 regularizar')
    train_setting.add_argument('--learning_rate', type=float, default=0.0001,
                               help='learning rate of training')
    train_setting.add_argument('--min_lr', type=float, default=0.00001,
                               help='min learning rate of training')
    train_setting.add_argument("--lr_decay", default=0.8, type=float,
                               help='learning rate decay')
    train_setting.add_argument('--optimizer', type=str, default='Adam',
                               help='optimizer of training')
    train_setting.add_argument('--clip_value', type=float, default=0.0,
                               help='clip value of para loss')
    train_setting.add_argument('--MAXITER', type=int, default=50,
                               help='Max iterative epoches')
    train_setting.add_argument('--max_epoch_changelr',type=int,default=10,
                               help='change learning rate after this epoch')
    train_setting.add_argument('--changelr_interval', type=int, default=5,
                               help='change lr again ')
    train_setting.add_argument('--early_stopping',type=int,default=20,
                               help='stop model when accuarcy is not rising')


    return parser.parse_args()