#!/usr/bin/env python3
"""
Hypha DSL - Specification Layer

Core data structures representing the declarative specification of a Petri net.
These are built by the decorator API and used to instantiate runtime objects.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Any, Tuple
from enum import Enum, auto


class PlaceType(Enum):
    """Type of place storage mechanism"""
    BAG = auto()  # Unordered collection
    QUEUE = auto()  # FIFO ordered collection


@dataclass
class PlaceSpec:
    """Specification for a place in the Petri net"""
    name: str  # Local name only, e.g. "waiting"
    place_type: PlaceType
    handler: Optional[Callable] = None  # For IOPlaces
    state_factory: Optional[Callable] = None
    is_io_input: bool = False
    is_io_output: bool = False


@dataclass
class GuardSpec:
    """Specification for a guard function"""
    func: Callable  # Signature: (tokens, bb, timebase) -> Iterator[tuple]


@dataclass
class TransitionSpec:
    """Specification for a transition in the Petri net"""
    name: str  # Local name only, e.g. "process_task"
    handler: Callable  # Signature: async (consumed, bb, timebase, [state]) -> yields
    guard: Optional[GuardSpec] = None
    state_factory: Optional[Callable] = None


@dataclass
class ArcSpec:
    """Specification for an arc connecting places and transitions"""
    source: 'PlaceRef | TransitionRef'  # Reference objects, not strings
    target: 'PlaceRef | TransitionRef'  # Reference objects, not strings
    weight: int = 1
    name: Optional[str] = None

    # Computed canonical parts for source/target (tuple of name components)
    source_parts: Tuple[str, ...] = field(init=False)
    target_parts: Tuple[str, ...] = field(init=False)

    def __post_init__(self):
        # Helper to extract parts from a ref/string/tuple
        def _parts_of(obj) -> Tuple[str, ...]:
            try:
                parts = obj.get_parts()
                return tuple(parts)
            except Exception:
                # obj might be a dotted string or a tuple/list
                if isinstance(obj, str):
                    return tuple(obj.split('.'))
                if isinstance(obj, (list, tuple)):
                    return tuple(obj)
                # Fallback: stringify
                return tuple(str(obj).split('.'))

        self.source_parts = _parts_of(self.source)
        self.target_parts = _parts_of(self.target)


@dataclass
class NetSpec:
    """Complete specification of a Petri net"""
    name: str  # Local name only, e.g. "SimpleProcessor"
    parent: Optional['NetSpec'] = None  # Parent spec for hierarchy
    places: Dict[str, PlaceSpec] = field(default_factory=dict)  # Local names as keys
    transitions: Dict[str, TransitionSpec] = field(default_factory=dict)  # Local names as keys
    arcs: List[ArcSpec] = field(default_factory=list)
    subnets: Dict[str, 'NetSpec'] = field(default_factory=dict)  # Local names as keys
    
    def get_fqn(self, local_name: Optional[str] = None) -> str:
        """Compute fully qualified name by walking parent chain"""
        return '.'.join(self.get_parts(local_name))

    def get_parts(self, local_name: Optional[str] = None) -> List[str]:
        """Return the fully qualified name as a list of path components.

        If local_name is provided, returns the parts for that member
        under this spec (e.g. ['Parent', 'Spec', 'local']). Otherwise
        returns the parts for this spec itself.
        """
        parts: List[str] = []
        if self.parent:
            parts.extend(self.parent.get_parts())

        parts.append(self.name)

        if local_name is not None:
            parts.append(local_name)

        return parts
    
    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of the net"""
        lines = ["graph TD"]
        
        def add_subnet(spec: NetSpec, indent: str = "    "):
            spec_fqn = spec.get_fqn()
            
            if spec.subnets:
                for subnet_name, subnet_spec in spec.subnets.items():
                    subnet_fqn = subnet_spec.get_fqn()
                    lines.append(f"{indent}subgraph {subnet_fqn}")
                    add_subnet(subnet_spec, indent + "    ")
                    lines.append(f"{indent}end")
            
            for place_name, place_spec in spec.places.items():
                place_fqn = spec.get_fqn(place_name)
                shape = "((" 
                close = "))" 
                prefix = "[INPUT]</br>" if place_spec.is_io_input else "[OUTPUT]</br>" if place_spec.is_io_output else "" 
                lines.append(f"{indent}{place_fqn}{shape}\"{prefix}{place_fqn}\"{close}")
            
            for trans_name in spec.transitions.keys():
                trans_fqn = spec.get_fqn(trans_name)
                lines.append(f"{indent}{trans_fqn}[{trans_fqn}]")
            
            for arc in spec.arcs:
                source_fqn = arc.source.get_fqn()
                target_fqn = arc.target.get_fqn()
                weight_label = f"|weight={arc.weight}|" if arc.weight > 1 else ""
                lines.append(f"{indent}{source_fqn} -->{weight_label} {target_fqn}")
        
        add_subnet(self)
        
        return "\n".join(lines)


class PlaceRef:
    """Reference to a place for use in arc definitions"""
    def __init__(self, local_name: str, parent_spec: NetSpec):
        self.local_name = local_name
        self.parent_spec = parent_spec
    
    def get_fqn(self) -> str:
        """Compute FQN dynamically"""
        return '.'.join(self.get_parts())
    
    # For backward compatibility
    @property
    def fqn(self) -> str:
        return self.get_fqn()

    def get_parts(self) -> List[str]:
        """Return the FQN as list of parts for this place ref."""
        return self.parent_spec.get_parts(self.local_name)

    @property
    def parts(self) -> List[str]:
        return self.get_parts()
    
    def __repr__(self):
        return f"PlaceRef({self.get_fqn()})"


class TransitionRef:
    """Reference to a transition for use in arc definitions"""
    def __init__(self, local_name: str, parent_spec: NetSpec):
        self.local_name = local_name
        self.parent_spec = parent_spec
    
    def get_fqn(self) -> str:
        """Compute FQN dynamically"""
        return '.'.join(self.get_parts())
    # For backward compatibility
    @property
    def fqn(self) -> str:
        return self.get_fqn()
    
    def __repr__(self):
        return f"TransitionRef({self.get_fqn()})"

    def get_parts(self) -> List[str]:
        return self.parent_spec.get_parts(self.local_name)

    @property
    def parts(self) -> List[str]:
        return self.get_parts()


class SubnetRef:
    """Reference to a subnet instance for accessing its places/transitions"""
    def __init__(self, spec: NetSpec):
        self.spec = spec
    
    def get_fqn(self) -> str:
        """Compute FQN dynamically"""
        return '.'.join(self.get_parts())
    
    # For backward compatibility
    @property
    def fqn(self) -> str:
        return self.get_fqn()
    
    def __getattr__(self, name: str):
        # Check places
        if name in self.spec.places:
            return PlaceRef(name, self.spec)
        
        # Check transitions
        if name in self.spec.transitions:
            return TransitionRef(name, self.spec)
        
        # Check subnets
        if name in self.spec.subnets:
            return SubnetRef(self.spec.subnets[name])
        
        raise AttributeError(f"No place, transition, or subnet named {name} in {self.get_fqn()}")
    
    def __repr__(self):
        return f"SubnetRef({self.get_fqn()})"

    def get_parts(self) -> List[str]:
        return self.spec.get_parts()

    @property
    def parts(self) -> List[str]:
        return self.get_parts()