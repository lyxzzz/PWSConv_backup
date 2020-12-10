from model_cfgs import ssd_cfg
from model_cfgs import stdn_cfg
from model_cfgs import tpn_cfg
# from model_cfgs.anchors_cfg import ANCHOR_CFG

def _update_cfgs(CFG, user_json_cfgs):
    for k in user_json_cfgs.keys():
        if k == 'type':
            continue
        if hasattr(CFG, k):
            setattr(CFG, k, user_json_cfgs[k])

class emptyCFG:
    json_cfg={}
    anchors_cfg = None


def get_cfgs(json_cfg):
    print("================model cfg================")
    print('backbone: {}'.format(json_cfg['backbone']['type']))
    print('header: {}'.format(json_cfg['header']['type']))
    if 'losses' in json_cfg:
        print('losses: {}'.format(json_cfg['losses']['type']))
    print('postprocessing: {}'.format(json_cfg['postprocessing']['type']))
    print("=========================================")
    

