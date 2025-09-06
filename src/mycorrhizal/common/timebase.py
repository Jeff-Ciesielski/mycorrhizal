
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from time import monotonic


class Timebase(ABC):
    @abstractmethod
    def now(self) -> float:
        pass
    
    def advance(self):
        pass

    def reset(self):
        pass
    
    def set(self, val: float):
        pass

class WallClock(Timebase):
    def now(self) -> float:
        return datetime.now().timestamp()
    
class UTCClock(Timebase):
    def now(self) -> float:
        return datetime.now(timezone.utc).timestamp()

class MonotonicClock(Timebase):
    def now(self) -> float:
        return monotonic()
    
class CycleClock(Timebase):
    def __init__(self, stepsize: float = 1):
        super().__init__
        self.cycles: float = 0
        self._stepsize = stepsize
        
    def now(self) -> float:
        return self.cycles
        
    def advance(self):
        self.cycles += self._stepsize
        
    def reset(self):
        self.cycles = 0
        
class DictatedClock(Timebase):
    def __init__(self, initial: float):
        self.value = initial
        self._initial = initial

    def now(self) -> float:
        return self.value
    
    def reset(self):
        self.value = self._initial

    def set(self, val: float):
        self.value = val
        