import time
import datetime

class TimeSpans:
    def __init__(self):
        self.starttime = None
        self.endtime = None
    def start(self):
        self.starttime = datetime.datetime.now()
    def end(self):
        self.endtime = datetime.datetime.now()
    def get_formatted_output(self, unit='ms'):
        delta = self.get_timespan(unit)
        return str(delta) + ' ' + unit
        
    def get_timespan(self, unit='ms'):
        assert self.starttime is not None, 'Start time is not set'
        assert self.endtime is not None, 'End time is not set'
        delta = self.endtime - self.starttime
        if unit == 'ms':
            return delta.total_seconds() * 1000
        elif unit == 's':
            return delta.total_seconds()
        elif unit == 'm':
            return delta.total_seconds() / 60
        elif unit == 'h':
            return delta.total_seconds() / 3600
        elif unit == 'd':
            return delta.total_seconds() / 86400
        else:
            raise ValueError('Invalid unit')

class TimeTracker:
    def __init__(self):
        self.timestamps = {}
        
    def start(self, name):
        assert name not in self.timestamps, 'Timestamp already exists'
        self.timestamps[name] = TimeSpans()
        self.timestamps[name].start()
        
    def end(self, name):
        assert name in self.timestamps, 'Timestamp does not exist'
        self.timestamps[name].end()
        print(name + ': ' + self.get_formatted_output(name))
        
    def get_timespan(self, name, unit='ms'):
        assert name in self.timestamps, 'Timestamp does not exist'
        return self.timestamps[name].get_timespan(unit)
    
    def get_formatted_output(self, name, unit='ms'):
        assert name in self.timestamps, 'Timestamp does not exist'
        return self.timestamps[name].get_formatted_output(unit)
        
    def print_all(self, unit='ms'):
        for name in self.timestamps:
            print(name + ': ' + self.get_formatted_output(name, unit))
        
        
if __name__ == '__main__':
    tt = TimeTracker()
    tt.start('test')
    time.sleep(1)
    tt.end('test')
    print(tt.get_timespan('test', 's'))
    print(tt.get_formatted_output('test', 's'))
    tt.print_all('s')
    tt.print_all('ms')


