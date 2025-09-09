from abc import ABC, abstractmethod
from datetime import datetime, timezone
from time import monotonic
import asyncio


class Timebase(ABC):
    @abstractmethod
    def now(self) -> float:
        pass

    @abstractmethod
    async def sleep(self, duration: float):
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

    async def sleep(self, duration: float):
        await asyncio.sleep(duration)


class UTCClock(Timebase):
    def now(self) -> float:
        return datetime.now(timezone.utc).timestamp()

    async def sleep(self, duration: float):
        await asyncio.sleep(duration)


class MonotonicClock(Timebase):
    def now(self) -> float:
        return monotonic()

    async def sleep(self, duration: float):
        await asyncio.sleep(duration)


class CycleClock(Timebase):
    def __init__(self, stepsize: float = 1):
        super().__init__()
        self.cycles: float = 0
        self._stepsize = stepsize
        self._advance_evt = asyncio.Event()

    async def sleep(self, duration: float):
        target = self.cycles + duration
        while self.cycles < target:
            await self._advance_evt.wait()

    def now(self) -> float:
        return self.cycles

    def advance(self):
        self.cycles += self._stepsize

        # Pulse the event to wake up sleepers
        self._advance_evt.set()
        self._advance_evt.clear()

    def reset(self):
        self.cycles = 0


class DictatedClock(Timebase):
    def __init__(self, initial: float):
        super().__init__()
        self.value = initial
        self._initial = initial
        self._set_evt = asyncio.Event()

    async def sleep(self, duration: float):
        target = self.value + duration
        while self.value < target:
            await self._set_evt.wait()

    def now(self) -> float:
        return self.value

    def reset(self):
        self.value = self._initial

    def set(self, val: float):
        self.value = val

        # Pulse the event to wake up sleepers
        self._set_evt.set()
        self._set_evt.clear()
