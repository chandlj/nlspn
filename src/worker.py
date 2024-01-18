import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import args as args_config
from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp
import torchvision.transforms.functional as TF

def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume

    return new_args

def worker(queue: mp.Queue, sender: Connection, env=None):
    if env is not None:
        os.environ.update(env)

    os.chdir(os.path.dirname(__file__))

    from model import get as get_model

    try:

        args = check_args(args_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network
        model = get_model(args)
        net = model(args)
        net.to(device)

        checkpoint = torch.load("pretrained/kittidc.pt")
        net.load_state_dict(checkpoint['net'])

        net.eval()
        rgb: torch.Tensor
        depth: torch.Tensor
        result: torch.Tensor
        complete: torch.Tensor

        while True:
            queue_item = queue.get()
            if queue_item == "STOP":
                break
            
            rgb, depth, result, complete = queue_item
            rgb = rgb.to(device)
            depth = depth.to(device)

            _, H, W = rgb.shape
            rgb = TF.normalize(
                rgb,
                (0.3134132, 0.34297643, 0.37416015),
                (0.20625564, 0.21985089, 0.25681712),
                inplace=True,
            )

            depth = TF.crop(
                depth, args.top_crop, 0, H - args.top_crop, W
            )

            sample = {"rgb": rgb[None], "dep": depth[None]}

            with torch.no_grad():
                out: dict[str, torch.Tensor] = net(sample)

            out = out["pred"].squeeze().detach().cpu()
            result[args.top_crop:, :] = out[:]
            # Invert and write to same buffer
            torch.logical_not(complete, out=complete)

            # Clear references to free memory
            del result
            del rgb
            del depth
            del out
            del complete
            torch.cuda.empty_cache()
    except Exception as e:
        # Send error to parent process
        sender.send(True)
        raise e
    finally:
        sender.close()