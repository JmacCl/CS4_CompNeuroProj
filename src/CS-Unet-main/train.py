import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.vision_transformer import CS_Unet as ViT_seg
from trainer import trainer_synapse
from trainer_ACDC import trainer_acdc
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse', help='root dir for data')  # '../data/processed_acdc_dataset'
parser.add_argument('--volume_path', type=str,  # ../data/processed_acdc_dataset
                    default='../data/Synapse', help='test dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str,
                    default='./results', help='output dir')
parser.add_argument('--test_save_dir', default='./results/acdc/predictions', help='saving prediction as nii!')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument("--n_skip", type=int,default=5)
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training_utils')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--base_weight', type=float,  default=0.0005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='inputs patch size of network inputs')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'ACDC': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 4,
            'z_spacing': 10,
        },
        'BraTS2020': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 4,
            'z_spacing': 10,
        },
    }

    # if args.batch_size != 24 and args.batch_size % 6 == 0:
    #     args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    # net.load_from(config)
    if args.dataset == "Synapse":
        trainer = {'Synapse': trainer_synapse,}
        trainer[dataset_name](args, net, args.output_dir)
    elif args.data == "BraTS2020":
        trainer = {'BraTS2020': trainer_brats2020, }
        trainer[dataset_name](args, net, args.output_dir)
    else:
        trainer_ACDC = {'ACDC': trainer_acdc,}
        trainer_ACDC[dataset_name](args, net, args.output_dir)

