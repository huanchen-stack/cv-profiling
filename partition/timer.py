import time

class Clock(object):

    def __init__(self):
        super().__init__()
        self.agg = 0
        self.start = None
        self.diff = None

    def tic(self):
        self.start = time.time()
    
    def toc(self):
        self.diff = time.time() - self.start
        self.agg += self.diff
    
    def get_time(self):
        return self.diff

    def get_agg(self):
        return self.agg