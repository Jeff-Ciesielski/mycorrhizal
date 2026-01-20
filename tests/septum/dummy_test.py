
import sys
sys.path.insert(0, "src")

from mycorrhizal.septum.core import septum, get_all_states

@septum.state()
def StateA():
    @septum.on_state
    async def on_state(ctx):
        return None
    @septum.transitions
    def transitions():
        return []

print(f"__name__ = {__name__}")
print(f"StateA.__module__ = {StateA.__module__}")

states = get_all_states()
for name in states:
    print(f"Registered: {name}")
