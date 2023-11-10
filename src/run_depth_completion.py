import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing import shared_memory
from typing import Tuple

# # Minimize randomness
# torch.manual_seed(args_config.seed)
# np.random.seed(args_config.seed)
# random.seed(args_config.seed)
# torch.cuda.manual_seed_all(args_config.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def run_depth_completion(receiver: Connection, pid: int, image_shape: Tuple[int, ...], depth_shape: Tuple[int, ...]):
    import sys, os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)

    os.chdir("nlspn/src")
    sys.path.append(os.getcwd())

    def active_virtualenv(exec_path):
        """
        copy virtualenv's activate_this.py
        exec_path: the python.exe path from sys.executable
        """
        # set env. var. PATH
        old_os_path = os.environ.get('PATH', '')
        os.environ['PATH'] = os.path.dirname(os.path.abspath(exec_path)) + os.pathsep + old_os_path
        base = os.path.dirname(os.path.dirname(os.path.abspath(exec_path)))
        # site-pachages path
        if sys.platform == 'win32':
            site_packages = os.path.join(base, 'Lib', 'site-packages')
        else:
            site_packages = os.path.join(base, 'lib', 'python%s' % sys.version[:3], 'site-packages')
        # modify sys.path
        prev_sys_path = list(sys.path)
        import site
        site.addsitedir(site_packages)
        sys.real_prefix = sys.prefix
        sys.prefix = base
        # Move the added items to the front of the path:
        new_sys_path = []
        for item in list(sys.path):
            if item not in prev_sys_path:
                new_sys_path.append(item)
                sys.path.remove(item)
        sys.path[:0] = new_sys_path
        return None

    active_virtualenv(sys.executable)

    try:
        import numpy as np
        import torch
        import torchvision.transforms.functional as TF

        from model import get as get_model
        from config import args as args_config

        image_shm = shared_memory.SharedMemory(name=f"image_{pid}")
        depth_shm = shared_memory.SharedMemory(name=f"depth_{pid}")
        result_shm = shared_memory.SharedMemory(name=f"result_{pid}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(args_config)
        net = model(args_config)
        net.to(device)

        checkpoint = torch.load(f"{os.getcwd()}/../results/NLSPN_KITTI_DC.pt")
        net.load_state_dict(checkpoint["net"])
        net.eval()

        rgb = np.ndarray(image_shape, dtype=np.uint8, buffer=image_shm.buf)
        depth = np.ndarray(depth_shape, dtype=np.float32, buffer=depth_shm.buf)
        result = np.ndarray(depth_shape, dtype=np.float32, buffer=result_shm.buf)

        receiver.send(True)
    
        while True:
            message = receiver.recv()
            if message == "TERMINATE":
                break
            elif message == True:
                pass
            else:
                raise Exception("Unexpected message")

            rgb_tensor = TF.to_tensor(rgb).to(device)
            _, H, W = rgb_tensor.shape
            rgb_tensor = TF.crop(rgb_tensor, args_config.top_crop, 0, H - args_config.top_crop, W)
            rgb_tensor = TF.normalize(
                rgb_tensor,
                (0.3134132, 0.34297643, 0.37416015),
                (0.20625564, 0.21985089, 0.25681712),
                inplace=True,
            )

            depth_tensor = TF.to_tensor(depth).to(device)
            depth_tensor = TF.crop(depth_tensor, args_config.top_crop, 0, H - args_config.top_crop, W)

            sample = {"rgb": rgb_tensor[None], "dep": depth_tensor[None]}
            out = net(sample)
            predicted = out["pred"][0].detach().cpu().numpy()

            result[args_config.top_crop:, :] = predicted[:]
            
            del rgb_tensor
            del depth_tensor
            del out
            torch.cuda.empty_cache()
            receiver.send(True)
    except Exception as e:
        receiver.send(f"ERROR: {e}")
        raise e
    finally:
        # Release shared memory from this process
        image_shm.close()
        depth_shm.close()
        result_shm.close()

