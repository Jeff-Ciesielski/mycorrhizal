"""
Cordyceps Petri Net Utilities: Parametric Fork and Join Interfaces
"""
from cordyceps.core import Interface, Place, Transition, Arc
from typing import Type


def Fork(n: int) -> Type[Interface]:
    """
    Returns an Interface subclass with 1 input place, n output places, and a transition
    that replicates each input token to all outputs by default.
    """
    # Build class attributes dict
    attrs = {}
    # Input place
    class Input(Place):
        pass
    attrs['Input'] = Input
    discovered_places = [Input]
    # Output places
    for i in range(n):
        output_cls = type(f'Output_{i+1}', (Place,), {})
        attrs[f'Output_{i+1}'] = output_cls
        discovered_places.append(output_cls)
    # Transition
    def input_arcs(self):
        return {'input': Arc(Input)}
    def output_arcs(self):
        return {f'output_{i+1}': Arc(attrs[f"Output_{i+1}"]) for i in range(n)}
    async def guard(self, pending):
        for token in pending['input']:
            self.consume('input', token)
        return bool(self._to_consume.get('input'))
    async def on_fire(self, consumed):
        for token in consumed['input']:
            for i in range(n):
                self.produce(f'output_{i+1}', token)
    transition_class_name = f'ForkTransition{n}'
    ForkTransition = type(transition_class_name, (Transition,), {
        'input_arcs': input_arcs,
        'output_arcs': output_arcs,
        'guard': guard,
        'on_fire': on_fire
    })
    attrs[transition_class_name] = ForkTransition
    discovered_transitions = [ForkTransition]
    attrs['_discovered_places'] = discovered_places
    attrs['_discovered_transitions'] = discovered_transitions
    clsname = f'Fork{n}'
    return type(clsname, (Interface,), attrs)


def Join(n: int) -> Type[Interface]:
    """
    Returns an Interface subclass with n input places, 1 output place, and a transition
    that moves all tokens from all inputs to the output.
    """
    # Build class attributes dict
    attrs = {}
    discovered_places = []
    # Input places
    for i in range(n):
        input_cls = type(f'Input_{i+1}', (Place,), {})
        attrs[f'Input_{i+1}'] = input_cls
        discovered_places.append(input_cls)
    # Output place
    class Output(Place):
        pass
    attrs['Output'] = Output
    discovered_places.append(Output)
    # Transition
    def input_arcs(self):
        return {f'input_{i+1}': Arc(attrs[f"Input_{i+1}"]) for i in range(n)}
    def output_arcs(self):
        return {'output': Arc(Output)}
    async def guard(self, pending):
        for label, tokens in pending.items():
            for token in tokens:
                self.consume(label, token)
        return any(self._to_consume.values())
    async def on_fire(self, consumed):
        for tokens in consumed.values():
            for token in tokens:
                self.produce('output', token)
    transition_class_name = f'JoinTransition{n}'
    JoinTransition = type(transition_class_name, (Transition,), {
        'input_arcs': input_arcs,
        'output_arcs': output_arcs,
        'guard': guard,
        'on_fire': on_fire
    })
    attrs[transition_class_name] = JoinTransition
    discovered_transitions = [JoinTransition]
    attrs['_discovered_places'] = discovered_places
    attrs['_discovered_transitions'] = discovered_transitions
    clsname = f'Join{n}'
    return type(clsname, (Interface,), attrs)
