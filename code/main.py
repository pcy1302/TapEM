import argparse
import numpy as np
import random
np.random.seed(0)
random.seed(0)


def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='TapEM')

    parser.add_argument('--embedder', nargs='?', default='TapEM',
                        help='Choose an embedder.')

    parser.add_argument('--embed_dim', type = int, default = 128,
                       help = 'embed dimension of author and paper')

    parser.add_argument('--model_path', type=str, default='../saved_model',
                       help='path to save model')

    parser.add_argument('--window', type = int, default = 3,
                       help = 'window size for indirect relation')

    parser.add_argument('--ctx_window', type=int, default=3,
                        help='window size for context')

    parser.add_argument('--c_len', type = int, default = 100,
                       help = 'max len of paper content')

    parser.add_argument('--batch_size', type = int, default = 500,
                       help = 'batch size of training')

    parser.add_argument('--learn_rate', type = float, default = 0.001,
                       help = 'learning rate')

    parser.add_argument('--train_iter_max', type = int, default = 1000,
                       help = 'max number of training iterations')

    parser.add_argument('--c_reg', type = float, default = 0.001,
                       help = 'coefficient of regularization')

    parser.add_argument('--margin_d', type = float, default = 0.1,
                       help = 'margin distance of augmented component')

    parser.add_argument('--c_tradeoff', type = float, default = 0.1,
                       help = 'tradeoff coefficient of augmented component')

    parser.add_argument('--init_std', type=float, default=0.1,
                        help='std for initialization for author embedding/predictor/edgeMLP')

    parser.add_argument('--data_path', type=str, default='../data',
                       help='path to data')

    parser.add_argument('--top_K', nargs='?', default = '1,2,5,10,20',
                       help='length of return list per paper in evaluation')

    parser.add_argument('--gpu_num', type= int, default=7,
                       help='gpu_num')

    parser.add_argument('--num_dir_neg', type=int, default=1,
                        help='num negative samples for metric learning')

    parser.add_argument('--num_indir_neg', type=int, default=1,
                        help='num negative samples for skipgram')

    parser.add_argument('--num_ctx_neg', type=int, default=1,
                        help='num_negatives')

    parser.add_argument('--dnn_dims', nargs='?', default='100,100',
                            help='Comma-separated layer dimensions for deep embedding '
                            'transformation.')

    parser.add_argument('--metric', nargs='?', help='score metric. L2 or dot', default='dot')

    parser.add_argument('--reg_semi', type=float, default=1.0,
                        help='coefficient of regularization')

    parser.add_argument('--reg_ctx', type=float, default=0.1,
                        help='coefficient of regularization')

    parser.add_argument('--reg_ml', type=float, default=0.01,
                        help='coefficient of regularization')

    parser.add_argument('--early_stop', type=int, default=20,
                        help='early_stop')

    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='Pretrain with camel')

    parser.add_argument('--ctx_ratio', type=float, default=0.01,
                        help='ctx_ratio')

    parser.add_argument('--save', action='store_true', default=False,
                        help='Save the best model')

    parser.add_argument('--dropout', type=float, default=0.15,
                        help='dropout')

    parser.add_argument('--rescale_grad', action='store_true', default=False,
                        help='rescale gradient')

    parser.add_argument('--grad_norm', type=float, default=10,
                        help='norm clipping')

    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='scheduler_factor')

    parser.add_argument('--scheduler_patience', type=float, default=5,
                        help='scheduler_patience')

    parser.add_argument('--scheduler_threshold', type=float, default=0.001,
                        help='scheduler_threshold')

    parser.add_argument('--version', nargs='?', default='top',
                        help='version (top or all)')

    parser.add_argument('--year', type=int, default=2013,
                        help='year (2013 or 2014)')

    return parser.parse_known_args()


def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)


def main():
    args, unknown = parse_args()
    print("=====Configurations=====")
    print(args)
    # printConfig(args)

    if args.embedder == 'camel':
        from camel import camel
        embedder = camel(args)

    elif args.embedder == 'TapEM':
        from TapEM import TapEM
        embedder = TapEM(args)


    embedder.training()



if __name__ == '__main__':
    main()
