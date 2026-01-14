# Mycorrhizal Examples

This directory contains examples demonstrating the three DSL systems and their features.

## Quick Start

Run any example:
```bash
uv run python examples/<module>/<example_name>.py
```

For example:
```bash
uv run python examples/hypha/minimal_hypha_demo.py
```

## Directory Structure

### [`hypha/`](./hypha/)
Petri Net examples using the Hypha DSL
- **`minimal_hypha_demo.py`** - Simple Petri net with basic concepts
- **`hypha_demo.py`** - Full-featured Petri net with subnets, IO places, error handling
- **`programmatic_hypha_demo.py`** - Building nets programmatically without decorators

### [`rhizomorph/`](./rhizomorph/)
Behavior Tree examples using the Rhizomorph DSL
- **`rhizomorph_example.py`** - Comprehensive behavior tree with threat detection
- **`rhizomorph_interface_demo.py`** - BT with interface-based access control

### [`enoki/`](./enoki/)
Finite State Machine examples using the Enoki DSL
- **`enoki_decorator_basic.py`** - Basic decorator-based state machine
- **`enoki_decorator_timeout.py`** - State machine with timeout handling

### [`interfaces/`](./interfaces/)
Interface system examples
- **`interface_dsl_demo.py`** - Using the @blackboard_interface decorator system

### [`spores/`](./spores/)
Spores event and object logging examples
- **`hypha_spores_example.py`** - Petri net with OCEL event logging for manufacturing workflow
- **`rhizomorph_spores_example.py`** - Behavior tree with event logging for threat assessment
- **`enoki_spores_example.py`** - State machine with lifecycle event logging for traffic light control

## What to Learn From Each Example

| If you want to learn... | Start with... |
|------------------------|---------------|
| Basic behavior trees | `rhizomorph/rhizomorph_example.py` |
| Basic Petri nets | `hypha/minimal_hypha_demo.py` |
| Basic state machines | `enoki/enoki_decorator_basic.py` |
| Interface basics | `interfaces/interface_dsl_demo.py` |
| Programmatic Petri nets | `hypha/programmatic_hypha_demo.py` |
| BT with interfaces | `rhizomorph/rhizomorph_interface_demo.py` |
| Full Petri net system | `hypha/hypha_demo.py` |
| Event logging with Hypha | `spores/hypha_spores_example.py` |
| Event logging with Rhizomorph | `spores/rhizomorph_spores_example.py` |
| Event logging with Enoki | `spores/enoki_spores_example.py` |

## Running All Examples

Test all examples:
```bash
# Hypha examples
for demo in examples/hypha/*.py; do
    echo "Testing $demo"
    uv run python "$demo"
done

# Rhizomorph examples
for demo in examples/rhizomorph/*.py; do
    echo "Testing $demo"
    uv run python "$demo"
done

# Enoki examples
for demo in examples/enoki/*.py; do
    echo "Testing $demo"
    uv run python "$demo"
done

# Spores examples
for demo in examples/spores/*.py; do
    echo "Testing $demo"
    uv run python "$demo"
done
```