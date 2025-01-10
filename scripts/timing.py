import tensorflow as tf
import time
from reproject import cuda_synchronize

class TFTimeRecord:
    def __init__(self) -> None:
        self.start_time = 0
        self.end_time = 0
    
    def start(self):
        self.start_time = time.perf_counter()
    
    def end(self):
        cuda_synchronize()
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        return elapsed


class Timing:
    def __init__(self) -> None:
        self.records = {}
        self.record_time = {}
        self.record_num = {}
        self.last_elapsed = {}
    
    def add(self, name):
        record = TFTimeRecord()
        self.records[name] = record
        self.record_time[name] = 0
        self.record_num[name] = 0
        self.last_elapsed[name] = 0
    
    def start(self, name):
        if name not in self.records:
            self.add(name)
        self.records[name].start()
    
    def end(self, name, exclude_from_time=False):
        if name not in self.records:
            print(f'Cannot .end(). Time record {name} does not exist!')
            return
        
        elapsed = self.records[name].end()
        if not exclude_from_time:
            self.record_time[name] += elapsed
            self.record_num[name] += 1
            self.last_elapsed[name] = elapsed
        
    def to_string(self, name=None):
        if name == None:
            ret = ''
            for i, name in enumerate(self.records):
                if i > 0:
                    ret += ', '
                ret += self.to_string(name)
            ret += f', Total {self.get_total_elapsed()*1000:.2f}/{self.get_total_time()*1000:.2f}ms'
        else:
            elapsed = self.last_elapsed[name]
            time = self.record_time[name]
            ret = f'{name} {elapsed*1000:.2f}/{time*1000:.2f}ms'
        return ret
    
    def get_total_elapsed(self):
        total = 0
        for t in self.last_elapsed.values():
            total += t
        return total

    def get_total_time(self):
        total = 0
        for t in self.record_time.values():
            total += t
        return total