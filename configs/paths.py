import socket
import getpass

username = getpass.getuser()
hostname = socket.gethostname()

dataroot = None
logroot = None
# BEHAVE dataset paths
behave_objs_path = None
behave_seqs_path = None
behave_calibs_path = None
# SMPL paths
smpl_assets_root = None
smpl_model_root = None

code = None

if hostname == 'helium-X550VX':
    pass
elif hostname == 'marvella':
    smpl_assets_root = "lib_smpl/assets"
    smpl_model_root = "/home/zahra/models/smpl/mano_v1_2/models" #"/BS/xxie2020/static00/mysmpl/smplh"
    code = "/home/zahra/workshop/hoi/interaction-learning/AutoSDF-code"
elif hostname == 'parham-laptop':
    pass
else:
    dataroot = 'data'
    logroot = 'logs'
    # BEHAVE dataset paths
    behave_objs_path = '/data/xiwang/behave/objects'
    behave_seqs_path = '/data/xiwang/behave/sequences'
    behave_calibs_path = '/data/xiwang/behave/calibs'
    smpl_assets_root = "lib_smpl/assets"
    smpl_model_root= "/local/home/aruzzi/interaction-learning/AutoSDF-code/saved_ckpt/mano_v1_2/models"
    code = "/local/aruzzi/interaction-learning/AutoSDF-code" # path to your project main folder

