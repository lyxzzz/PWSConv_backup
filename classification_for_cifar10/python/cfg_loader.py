from model_cfgs import cls_cfg
# from model_cfgs.anchors_cfg import ANCHOR_CFG

cfg_dict = {
    'cls':cls_cfg
}

def _update_cfgs(CFG, user_json_cfgs):
    for k in user_json_cfgs.keys():
        if k == 'type':
            continue
        if hasattr(CFG, k):
            setattr(CFG, k, user_json_cfgs[k])

class emptyCFG:
    json_cfg={}
    anchors_cfg = None


def get_cfgs(cfg_name, json_cfg):
    if cfg_name not in cfg_dict:
        ROOT_CFG = emptyCFG
        ROOT_CFG.json_cfg = json_cfg
    else:
        ROOT_CFG = emptyCFG
        ROOT_CFG.json_cfg = json_cfg
        ROOT_CFG.data_cfg = cfg_dict[cfg_name].DATA_CONFIG

    print("================model cfg================")
    print('backbone: {}'.format(json_cfg['backbone']['type']))
    if 'losses' in json_cfg:
        print('losses: {}'.format(json_cfg['losses']['type']))
    print("=========================================")
    # if 'postprocessing' in json_cfg:
    #     _update_cfgs(ROOT_CFG.POST_PROCESSING_CONFIG, json_cfg['postprocessing'])
    if 'test_parameters' in json_cfg:
        _update_cfgs(ROOT_CFG.data_cfg, json_cfg['test_parameters'])
    if 'train_parameters' in json_cfg:
        _update_cfgs(ROOT_CFG.data_cfg, json_cfg['train_parameters'])
    return ROOT_CFG
    

