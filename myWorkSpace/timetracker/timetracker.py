import time
import datetime

class TimeSpans:
    def __init__(self,name=None):
        self.starttime = None
        self.endtime = None
        self.unit = 'ms'
        self.name = name
        
    def set_unit(self, unit):
        self.unit = unit
    def set_name(self, name):
        self.name = name
    
    @staticmethod
    def fomart(time, fmt = '%H:%M:%S'):
        return time.strftime(fmt)
    
    def start(self,print = False):
        self.starttime = datetime.datetime.now()
        if print:
            print(self.name + ' Start: ' + TimeSpans.fomart(self.starttime))
        return self.name + ' Start: ' + TimeSpans.fomart(self.starttime)

    def end(self,print = False):
        self.endtime = datetime.datetime.now()
        if print:
            print(self.name + ' End: ' + TimeSpans.fomart(self.endtime))
        return self.name + ' End: ' + TimeSpans.fomart(self.endtime)
    
    def get_formatted_timespan(self, unit=None):
        assert self.name is not None, 'Name is not set'
        delta = self.get_timespan(unit)
        return self.name + ': ' + str(delta) + ' ' + unit
        
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
        self.timestamps[name] = TimeSpans(name)
        return self.timestamps[name].start()
        
    def end(self, name):
        assert name in self.timestamps, 'Timestamp does not exist'
        return self.timestamps[name].end()
        
    def get_timespan(self, name, unit='ms'):
        assert name in self.timestamps, 'Timestamp does not exist'
        return self.timestamps[name].get_timespan(unit)
    
    def get_formatted_timespan(self, name, unit='ms'):
        assert name in self.timestamps, 'Timestamp does not exist'
        return self.timestamps[name].get_formatted_timespan(unit)
        
    def get_all(self, unit='ms'):
        for name in self.timestamps:
           yield self.timestamps[name].get_timespan(unit)
        
        
if __name__ == '__main__':
    tt = TimeTracker()
    tt.start('test')
    time.sleep(1)
    tt.end('test')
    print(tt.get_timespan('test', 's'))
    print(tt.get_formatted_timespan('test', 's'))



