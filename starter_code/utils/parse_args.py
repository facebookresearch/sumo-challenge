import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='3D Scene Reconstruction')

    # Optimization arguments
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--optim', default='adagrad', help='Optimizer: sgd|adam|adagrad|adadelta|rmsprop')

    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')

    # Dataset
    parser.add_argument('--dataset', default='SUMO', help='Dataset name: sumo')
    parser.add_argument('--H', default=1024, type=int, help='')
    parser.add_argument('--W', default=1024, type=int, help='')
    parser.add_argument('--scene_W', default=1024, type=int, help='')
    parser.add_argument('--scene_H', default=1024, type=int, help='')
    parser.add_argument('--model_H', default=192, type=int, help='')
    parser.add_argument('--model_W', default=256, type=int, help='')

    # Model
    parser.add_argument('--NET', default='cvgg16', help='Network used')
    parser.add_argument('--EXP_TYPE', default='rec3d', help='Type of experiment - rec3d=3D reconstruction bb3d=3D bb detection bb2d=2D BB detection')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--resume', type=int, default=0, help='If 1, resume training')
    parser.add_argument('--npred', default=150, type=int, help='Number of object predictions')

    # Point cloud processing
    parser.add_argument('--pc_augm_scale', default=0, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=0, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=0, type=int, help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--code_nfts', default=1024, type=int, help='Encoder output feature size')
    parser.add_argument('--eval', default=0, type=int, help='If 1, evaluate using best model')
    parser.add_argument('--gpu_id', default='0', help='GPU ID')

    # Misc
    parser.add_argument('--show', default=0, type=int, help='If 1, display loaded data')

    # TopNet
    parser.add_argument('--outpts', default=8*2048, type=int, help='Number of output points generated for each object')
    parser.add_argument('--dist', default='nnd', help='Point Cloud distance used in training')

    args = parser.parse_args()
    args.start_epoch = 0

    return args
