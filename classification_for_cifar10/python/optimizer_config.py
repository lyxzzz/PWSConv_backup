import tensorflow as tf

def get_optimizer_from_cfg(learning_rate, optimizer_cfg):
    if optimizer_cfg == None:
        optimizer_cfg = {"type":"Momentum","momentum":0.9}
    print('***********************')
    print("apply {} optimizer".format(optimizer_cfg['type']))
    print('***********************')
    if optimizer_cfg['type'] == 'Momentum':
        opt = tf.train.MomentumOptimizer(learning_rate, optimizer_cfg.get('momentum', 0.9))
    elif optimizer_cfg['type'] == 'Rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate,
            decay=optimizer_cfg.get('rmsprop_decay', 0.9),
            momentum=optimizer_cfg.get('rmsprop_momentum', 0.9),
            epsilon=optimizer_cfg.get('opt_epsilon', 1.0))
    else:
        opt = tf.train.AdamOptimizer(learning_rate)
    
    return opt