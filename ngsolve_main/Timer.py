#Simple timer class for performence measurements
import time

class Timer:
    def __init__(self) -> None:
        self.start = 0.0
        
    
    def startTimer(self) -> None:
        self.start = time.perf_counter()
    
    def printTimer(self) -> float:
        end = time.perf_counter()
        res_time = end - self.start
        print("Elapsed time: {}s".format(res_time))
        return (res_time)