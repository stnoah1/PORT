import argparse

def get_parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('--dataset', default='panoptic', type=str, help='')
    parser.add_argument('--checkpoint', default='checkpoint/debug', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--work_dir', default='results', type=str, metavar='PATH', help='')

    # Evaluate choice
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')

    # Training detail
    parser.add_argument('--batch_size', default=2048, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of training epochs')

    # Occlusion Estimation
    parser.add_argument('--thd_percent_st', default=1, type=int, help='number of training epochs')
    parser.add_argument('--thd_percent_offset', default=20, type=int, help='number of training epochs')
    parser.add_argument('--thd_percent_stride', default=1, type=int, help='number of training epochs')

    # Optimizer
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight_decay')
    parser.add_argument('--lr', default=5.0e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)

    # Experimental setting
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--val_freq', default=3, type=int, help='validation frequency')

    # Apex
    parser.add_argument('--half', default=True, type=lambda x: (str(x).lower() == 'true'), help='apex')
    parser.add_argument('--ssl', default=False, type=lambda x: (str(x).lower() == 'true'), help='')
    parser.add_argument('--amp_opt_level', type=int, default=1)

    # Transformer
    parser.add_argument('--embd_dim', default=64 , type=int, help='')
    parser.add_argument('--depth', default=6, type=int, help='')
    parser.add_argument('--head', default=4, type=int, help='')
    parser.add_argument('--mlp_dim', default=128, type=int, help='')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--embd_dropout', default=0.1, type=float, help='embd_dropout rate')
    parser.add_argument('--mask_prob', default=0.3, type=float, help='mask_probability')
    parser.add_argument('--random_patch_prob', default=0., type=float, help='')
    parser.add_argument('--replace_prob', default=1., type=float, help='')
    parser.add_argument('--graph_conv', default=True, type=lambda x: (str(x).lower() == 'true'), help='')

    parser.add_argument('--model', default='cvt', type=str, help='model')
    parser.add_argument('--proj_kernels', nargs='+', default=[3], type=int, help='model')
    parser.add_argument('--self_supervision', default='simmim', type=str, help='self-supervised learning task')
    parser.add_argument('--lambda_2d', default=1., type=float, help='')
    parser.add_argument('--use_bone_2d', default=False, type=lambda x: (str(x).lower() == 'true'), help='use bone feature')
    parser.add_argument('--rand_rot', default=False, type=lambda x: (str(x).lower() == 'true'), help='random rotation')
    parser.add_argument('--rand_move', default=False, type=lambda x: (str(x).lower() == 'true'), help='random move')

    # data_path

    parser.add_argument('--train_ann_path', default="data/panoptic/annotations/panoptic_train.json", type=str, metavar='PATH', help='')
    parser.add_argument('--test_ann_path', default="data/panoptic/annotations/panoptic_test.json", type=str, metavar='PATH', help='')
    parser.add_argument('--cam_path', default=None, type=str, metavar='PATH', help='')
    parser.add_argument('--kpt2d_stat_path', default=None, type=str, metavar='PATH', help='')
    parser.add_argument('--kpt3d_stat_path', default=None, type=str, metavar='PATH', help='')
    parser.add_argument('--kpt2d_train_pred_path', default=None, type=str, metavar='PATH', help='')
    parser.add_argument('--kpt2d_test_pred_path', default=None, type=str, metavar='PATH', help='')

   # Architecture
    parser.add_argument('--shared_topology', default=False, type=lambda x: (str(x).lower() == 'true'), help='use_toplogy')

    return parser.parse_args()
