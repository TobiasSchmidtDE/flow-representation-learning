import argparse
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path    
import datetime
import json

from lib.aux import create_exp_dir
from lib.DiffPDE import DiffPDE
from lib.HJPDE import HJPDE
from lib.trainer_ot_scratch import TrainerOTScratch
from vae import ConvVAE


def main():
    """
    ===[ PDEs ]=========================================================================================
    -K, --num-support-sets     : set number of PDEs
    -D, --num-timesteps        : set number of timesteps
    --support-set-lr           : set learning rate for learning PDEs
    
    ===[ Training ]=================================================================================================
    --max-iter                 : set maximum number of training iterations
    --batch-size               : set training batch size
    --log-freq                 : set number iterations per log
    --ckp-freq                 : set number iterations per checkpoint model saving
    --tensorboard              : use TensorBoard
    
    ===[ CUDA ]=====================================================================================================
    --cuda                     : use CUDA during training (default)
    --no-cuda                  : do NOT use CUDA during training
    ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="Latent Flow training script")
    # === PDEs ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, help="set number of PDEs", default=3)
    parser.add_argument('-D', '--num-timesteps', type=int, help="set number of timesteps", default=16)
    parser.add_argument('--support-set-lr', type=float, default=1e-4, help="set learning rate for learning PDEs")
    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=128, help="set batch size")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
    parser.add_argument("--shapes3d", type=bool, default=False)
    # === CUDA ======================================================================================================= #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=False)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Create output dir and save current arguments
    exp_dir = Path('./experiments/ot_scratch')
    # create subfolder with current timestamp and config args
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = exp_dir / '{}'.format(current_time)
    exp_dir.mkdir(parents=True, exist_ok=True)
    # save config args as dict
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f)
    

    # CUDA
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.shapes3d == True:
        G = ConvVAE(num_channel=3,latent_size=15*15+1,img_size=64)
        G.load_state_dict(torch.load("vae_shapes3d.pt", map_location='cpu'))
        print("Initialize Shapes3D VAE")
    else:
        G = ConvVAE(num_channel=3, latent_size=18*18, img_size=28)
        print("Intialize MNIST VAE")

    # Build PDEs
    print("#. Build Support Sets S...")
    print("  \\__Number of Support Sets    : {}".format(args.num_support_sets))
    print("  \\__Number of Timesteps       : {}".format(args.num_timesteps))
    print("  \\__Latent    Dimension       : {}".format(G.latent_size))

    S = HJPDE(num_support_sets=args.num_support_sets,
              num_timesteps=args.num_timesteps,
              support_vectors_dim=G.latent_size)

    S_Prior = DiffPDE(num_support_sets=args.num_support_sets,
              num_timesteps=args.num_timesteps,
              support_vectors_dim=G.latent_size)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))


    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S_Prior.parameters() if p.requires_grad)))

    # Set up trainer
    
    print("MNIST DATASET LOADING")
    #train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=data_config['batch_size'])
    dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
    #train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)],generator=torch.Generator(device='cuda'))
    #dataset = DSprites(root='/nfs/data_chaos/ysong/simplegan_experiments/dataset', transform=transforms.ToTensor())
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator(device=device))
    trn = TrainerOTScratch(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
                            data_loader=data_loader)

    # Train
    trn.train(generator=G, support_sets=S, prior = S_Prior)
    trn.eval(generator=G, support_sets=S, prior = S_Prior)


if __name__ == '__main__':
    main()
