import threading
import numpy as np
import time
import queue
import math

class DataThread (threading.Thread):
    def __init__(self, number, dataqueue, load_func, index_list):
        threading.Thread.__init__(self)
        self.threadNum = number
        self.dataqueue = dataqueue
        self.loadfunc = load_func
        self.indexlist = index_list
        self.epochComplete = False
        self.epochSupply = True
        self.run_signal = True
        self.now_epoch = 0

    def addTask(self, index_list):
        if self.epochComplete:
            if not self.epochSupply:
                self.indexlist = index_list
                self.now_epoch = self.now_epoch + 1
                self.epochSupply = True
                self.epochComplete = False
    
    def epochEnd(self):
        return self.epochComplete
    
    def close(self):
        self.run_signal = False

    def run(self):
        print ("thread[{}] start".format(self.threadNum))
        while self.run_signal:
            if self.epochComplete:
                time.sleep(2)
            else:
                index = 0
                length = len(self.indexlist)
                while index < length:
                    tmp_data = self.loadfunc(self.indexlist[index], self.now_epoch)
                    self.dataqueue.put(tmp_data)
                    index += 1
                self.epochComplete = True
                self.epochSupply = False

class dataPool(threading.Thread):
    def __init__(self, thread_num, batch_num, queue_size, data_size, item_size, load_func, epoch=1):
        threading.Thread.__init__(self)
        self.thread_num = thread_num
        self.thread_list = []
        self.batch_num = batch_num
        self.queue_size = queue_size
        self.data_size = data_size
        self.item_size = item_size
        self.run_signal = True

        if self.data_size % self.batch_num == 0:   
            self.total_batch_size = int(self.data_size / self.batch_num)
            self.last_peace_num = self.batch_num
        else:
            self.total_batch_size = int(self.data_size / self.batch_num) + 1
            self.last_peace_num = self.data_size % self.batch_num
        
        self.now_batch_size = 0

        self.load_func = load_func
        self.epoch = epoch
        self.now_epoch = 1
        self.index_array = np.arange(0, data_size)

        single_split = int(self.data_size / self.thread_num)
        self.index_split = [0]
        for i in range(1, self.thread_num):
            self.index_split.append(i * single_split)
        self.index_split.append(self.data_size)
        
        self.data_queue = queue.Queue(batch_num * queue_size)
        self.batch_queue = queue.Queue(queue_size)

        np.random.shuffle(self.index_array)
        for i in range(self.thread_num):
            tempThread = DataThread(i, self.data_queue, self.load_func, self.index_array[self.index_split[i]:self.index_split[i+1]])
            self.thread_list.append(tempThread)
        
        for thread in self.thread_list:
            thread.start()

    def run(self):
        now_batch_index = 1
        now_epoch_index = 0
        while self.run_signal:
            all_end = True
            for i in range(self.thread_num):
                if not self.thread_list[i].epochEnd():
                    all_end = False
            if all_end:
                if self.now_epoch < self.epoch:
                    np.random.shuffle(self.index_array)
                    for i in range(self.thread_num):
                        self.thread_list[i].addTask(self.index_array[self.index_split[i]:self.index_split[i+1]])
                    self.now_epoch += 1

            if now_epoch_index < self.epoch:
                if self.batch_queue.qsize() < self.queue_size:
                    if now_batch_index == self.total_batch_size:
                        pop_batch_num = self.last_peace_num
                    else:
                        pop_batch_num = self.batch_num
                    while self.data_queue.qsize() < pop_batch_num:
                        time.sleep(1)
                    batch_data = []
                    for i in range(self.item_size):
                        batch_data.append([])
                    for i in range(pop_batch_num):
                        single_data = self.data_queue.get_nowait()
                        for i in range(self.item_size):
                            batch_data[i].append(single_data[i])
                    self.batch_queue.put(batch_data)

                    now_batch_index += 1
                    if now_batch_index > self.total_batch_size:
                        now_batch_index = 1
                        now_epoch_index += 1
                else:
                    time.sleep(1)
    
    def getBatchData(self):
        if self.now_batch_size == self.total_batch_size:
            self.now_batch_size = 0
            return 0
        batch = self.batch_queue.get()
        self.now_batch_size += 1
        return batch
    
    def close(self):
        for i in range(self.thread_num):
            self.thread_list[i].close()
        for i in range(self.thread_num):
            self.thread_list[i].join()
        self.run_signal = False



