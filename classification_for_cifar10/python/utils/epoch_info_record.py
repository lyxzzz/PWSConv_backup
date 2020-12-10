import tensorflow as tf
import time
import numpy as np

from utils import eval_tools

def gethms(sectime):
    total_min = int(sectime / 60)
    total_sec = sectime - total_min * 60
    total_hour = int(total_min / 60)
    total_min = total_min - total_hour * 60
    return [total_hour, total_min, total_sec]

class EpochRecorder:
    def __init__(self, loss_dict, summary_writer, restore_epoch, max_epoch):
        self.loss_dict = loss_dict
        self.program_start = time.time()
        self.epoch_start = self.program_start
        self.acc_dict = ['top1err']
        self.summary_writer = summary_writer
        self.last_data_dict = {}
        self.restore_epoch = restore_epoch
        self.max_epoch = max_epoch - restore_epoch

    def start_epoch(self):
        self.epoch_start = time.time()

    def summary_epoch(self, loss_list, acc_list, learning_rate, epochs, steps, steps_per_epoch, scope='train'):
        print("***************************************{}***************************************".format(scope))
        if scope in self.last_data_dict:
            last_loss = self.last_data_dict[scope][0]
            last_acc = self.last_data_dict[scope][1]
        else:
            last_loss = np.zeros((len(self.loss_dict)), dtype=np.float32)
            last_acc = np.zeros((3), dtype=np.float32)
        loss_list = loss_list / steps_per_epoch

        acc_val = [eval_tools.top_error(acc_list)]
        summary_epoch = tf.Summary()
        summary_name = scope + "_condition/"
        loss_name = summary_name + "loss_"
        for index,name in enumerate(self.loss_dict):
            tag_name = loss_name + name
            summary_epoch.value.add(tag=tag_name, simple_value=loss_list[index])
        
        acc_name = summary_name + "acc_"

        for index,name in enumerate(self.acc_dict):
            tag_name = acc_name + name
            summary_epoch.value.add(tag=tag_name, simple_value=acc_val[index])
        
        self.summary_writer.add_summary(summary_epoch, global_step=epochs)

        now_time = time.time()
        total_time = now_time - self.program_start
        rest_time = total_time * (self.max_epoch - epochs) / (epochs - self.restore_epoch)

        total_hms = gethms(total_time)
        rest_hms = gethms(rest_time)

        epoch_time = now_time - self.epoch_start
        avg_time_per_step = epoch_time / steps_per_epoch

        print(scope, ' Epoch[{:03d}] Step[{:06d}],LR {:.6f}, {:.2f} seconds/step {:.2f} seconds, total: {}h {}m {:.2f}s, remain: {}h {}m {:.2f}s'\
            .format(epochs, steps, learning_rate, avg_time_per_step, epoch_time, total_hms[0], total_hms[1], total_hms[2], rest_hms[0], rest_hms[1], rest_hms[2]))
        print('\tloss condition',end='')
        for index,name in enumerate(self.loss_dict):
            print(", ", name, "[{:.4f} ~ {:.4f}]".format(last_loss[index], loss_list[index]), end='')
        print('\n\tacc condition',end='')

        for index,name in enumerate(self.acc_dict):
            print(", ", name, "[{:.4f} ~ {:.4f}]".format(last_acc[index], acc_val[index]), end='')
        print('\n',end='',flush=True)
        print("******************************************************************************")
        self.last_data_dict[scope] = [loss_list, acc_val]