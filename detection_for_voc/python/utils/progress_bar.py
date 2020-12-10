import math
import time
class ProgressBar:
    def __init__(self, slice, total_size):
        self.slice = int(math.ceil(slice / 25) * 25)
        self.total_size = total_size
        self.total_slice = [' '] * self.slice
        self.block_size = int(total_size / self.slice)
        self.first_print = True
        self.start_time = time.time()
    
    def print(self, now_index):
        now_rate = 1 + now_index // self.block_size
        for i in range(now_rate):
            if i < self.slice:
                self.total_slice[i] = '='
            else:
                break
        progress_rate = '['+ ''.join(self.total_slice) + ']'
        if self.first_print:
            self.start_time = time.time()
            self.first_print = False
        now_time = time.time()
        time_consume = now_time - self.start_time
        time_consume_min = int(time_consume/60)
        time_consume_sec = time_consume - 60 * time_consume_min
        time_last = (self.total_size - now_index) / now_index * time_consume
        time_last_min = int(time_last/60)
        time_last_sec = time_last - 60 * time_last_min
        time_str = ' use: {}m {:.2f}s remain: {}m {:.2f}s'.format(time_consume_min, time_consume_sec, time_last_min, time_last_sec)
        print('\r[{}/{} {:.2f}%]: '.format(now_index, self.total_size, 100*now_index/self.total_size) + progress_rate + time_str, end='')
        if now_index >= self.total_size:
            print('\ntotal_time: {}m {:.2f}s'.format(time_consume_min, time_consume_sec))

