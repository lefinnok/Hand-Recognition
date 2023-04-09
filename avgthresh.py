
class AverageThreshold:
    def __init__(self, sampleSize = 10):
        self.list = []
        self.thresh = 0
        self.sampleSize = sampleSize
    
    def put