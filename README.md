# Mycorrhizal

[![Tests](https://github.com/Jeff-Ciesielski/mycorrhizal/actions/workflows/test.yml/badge.svg)](https://github.com/Jeff-Ciesielski/mycorrhizal/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Jeff-Ciesielski/mycorrhizal/branch/main/graph/badge.svg)](https://codecov.io/gh/Jeff-Ciesielski/mycorrhizal)
[![Documentation Status](https://readthedocs.org/projects/mycorrhizal/badge/?version=latest)](https://mycorrhizal.readthedocs.io/en/latest/?badge=latest)
[![Python Version](https://img.shields.io/pypi/pyversions/mycorrhizal)](https://pypi.org/project/mycorrhizal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for building safe, structured, concurrent, event-driven systems.

## Overview

Mycorrhizal provides four domain-specific languages (DSLs) for modeling and implementing different aspects of complex systems:

* **Hypha** - Colored Petri nets for workflow modeling and orchestration
* **Rhizomorph** - Behavior trees for decision-making and control logic
* **Septum** - Finite state machines for stateful components
* **Spores** - Event and object logging for observability and process mining

Each DSL can be used independently or combined to build sophisticated systems. All modules share common infrastructure for state management (blackboards) and time abstraction, enabling seamless composition.

## Installation

```bash
pip install mycorrhizal
```

Requires Python 3.10 or later.

## Quick Start

### Behavior Tree (Rhizomorph)

Define behavior trees using a decorator-based DSL:

```python
from mycorrhizal.rhizomorph.core import bt, Runner, Status

@bt.tree
def ThreatResponse():
    @bt.action
    async def assess_threat(bb) -> Status:
        # Analyze threat level
        return Status.SUCCESS if bb.threat_level > 5 else Status.FAILURE

    @bt.action
    async def engage_countermeasures(bb) -> Status:
        # Respond to threat
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield assess_threat
        yield engage_countermeasures

# Run the behavior tree
runner = Runner(ThreatResponse, bb=blackboard)
await runner.tick_until_complete()
```

### Petri Net (Hypha)

Model workflows with colored Petri nets:

```python
from mycorrhizal.hypha.core import pn, Runner, PlaceType

@pn.net
def ProcessingNet(builder):
    # Define places
    pending = builder.place("pending", type=PlaceType.QUEUE)
    processed = builder.place("processed", type=PlaceType.QUEUE)

    # Define transitions
    @builder.transition()
    async def process(consumed, bb, timebase):
        for token in consumed:
            result = await handle(token)
            yield {processed: result}

    # Wire the net
    builder.arc(pending, process).arc(processed)

# Run the Petri net
runner = Runner(ProcessingNet, bb=blackboard)
await runner.start(timebase)
```

### Finite State Machine (Septum)

Build stateful components with FSMs:

```python
from mycorrhizal.septum.core import septum, StateMachine, LabeledTransition

@septum.state()
def IdleState():
    @septum.on_state
    async def on_state(ctx):
        if ctx.msg == "start":
            return Events.START
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.START, ProcessingState),
        ]

# Create and run the FSM
fsm = StateMachine(initial_state=IdleState, common_data={})
await fsm.initialize()
fsm.send_message("start")
await fsm.tick()
```

## Key Features

### Shared Infrastructure

All DSLs use common building blocks:

* **Blackboards** - Typed shared state using Pydantic models
* **Interfaces** - Decorator-based access control for blackboard fields
* **Timebase** - Abstract time for simulation and testing

### Composition

Combine DSLs to model complex systems:

* Embed behavior trees in Petri net transitions
* Run state machines within behavior tree actions
* Use Petri nets to orchestrate FSM-based components

### Observability

The Spores module provides OCEL-compliant logging:

* Automatic event extraction from DSL execution
* Object lifecycle tracking
* Transport layer for custom backends

## Examples

The repository contains comprehensive examples:

* `examples/hypha/` - Petri net patterns
* `examples/rhizomorph/` - Behavior tree patterns
* `examples/septum/` - State machine patterns
* `examples/spores/` - Event logging integration
* `examples/interfaces/` - Type-safe blackboard access

Run examples with:

```bash
uv run python examples/hypha/minimal_hypha_demo.py
```

See [examples/README.md](https://github.com/Jeff-Ciesielski/mycorrhizal/tree/main/examples/) for a complete guide.

## Visualization

All three DSLs support Mermaid diagram export for documentation and debugging. You can generate diagrams programmatically:

```python
# Behavior Tree
diagram = MyTree.to_mermaid()

# Petri Net
diagram = MyNet.to_mermaid()

# Finite State Machine
from mycorrhizal.septum.util import to_mermaid
diagram = to_mermaid(fsm)
```

### Example Diagrams

**Behavior Tree (Rhizomorph)** - A threat response system with decorators:

```mermaid
flowchart TD
  N1["Selector<br/>root"]
  N1 --> N2
  N2["Subtree<br/>Engage"]
  N2 --> N3
  N3["Sequence<br/>engage_threat"]
  N3 --> N4
  N4((CONDITION<br/>threat_detected))
  N3 --> N5
  N5["Decor<br/>Failer(Gate(cond=battery_ok)(Timeout(0.12s)(engage)))"]
  N5 --> N6
  N6["Decor<br/>Gate(cond=battery_ok)(Timeout(0.12s)(engage))"]
  N6 --> N7
  N7["Decor<br/>Timeout(0.12s)(engage)"]
  N7 --> N8
  N8((ACTION<br/>engage))
  N1 --> N9
  N9["Sequence<br/>patrol"]
  N9 --> N10
  N10((CONDITION<br/>has_waypoints))
  N9 --> N11
  N11((ACTION<br/>go_to_next))
  N9 --> N12
  N12["Decor<br/>Succeeder(Retry(3)(Timeout(1.0s)(scan_area)))"]
  N12 --> N13
  N13["Decor<br/>Retry(3)(Timeout(1.0s)(scan_area))"]
  N13 --> N14
  N14["Decor<br/>Timeout(1.0s)(scan_area)"]
  N14 --> N15
  N15((ACTION<br/>scan_area))
  N1 --> N16
  N16["Decor<br/>Failer(RateLimit(0.200000s)(telemetry_push))"]
  N16 --> N17
  N17["Decor<br/>RateLimit(0.200000s)(telemetry_push)"]
  N17 --> N18
  N18((ACTION<br/>telemetry_push))
```

**Petri Net (Hypha)** - A task processing system with error handling:

```mermaid
graph TD
    subgraph TaskProcessingSystem.TaskGen
        TaskProcessingSystem.TaskGen.source(("[INPUT]</br>TaskProcessingSystem.TaskGen.source"))
    end
    subgraph TaskProcessingSystem.TaskProc
        TaskProcessingSystem.TaskProc.input(("TaskProcessingSystem.TaskProc.input"))
        TaskProcessingSystem.TaskProc.processing(("TaskProcessingSystem.TaskProc.processing"))
        TaskProcessingSystem.TaskProc.completed(("TaskProcessingSystem.TaskProc.completed"))
        TaskProcessingSystem.TaskProc.failed(("TaskProcessingSystem.TaskProc.failed"))
        TaskProcessingSystem.TaskProc.take_to_processing[TaskProcessingSystem.TaskProc.take_to_processing]
        TaskProcessingSystem.TaskProc.do_processing[TaskProcessingSystem.TaskProc.do_processing]
        TaskProcessingSystem.TaskProc.input --> TaskProcessingSystem.TaskProc.take_to_processing
        TaskProcessingSystem.TaskProc.take_to_processing --> TaskProcessingSystem.TaskProc.processing
        TaskProcessingSystem.TaskProc.processing --> TaskProcessingSystem.TaskProc.do_processing
        TaskProcessingSystem.TaskProc.do_processing --> TaskProcessingSystem.TaskProc.completed
        TaskProcessingSystem.TaskProc.do_processing --> TaskProcessingSystem.TaskProc.failed
    end
    subgraph TaskProcessingSystem.Notify
        TaskProcessingSystem.Notify.input(("TaskProcessingSystem.Notify.input"))
        TaskProcessingSystem.Notify.email_sink(("[OUTPUT]</br>TaskProcessingSystem.Notify.email_sink"))
        TaskProcessingSystem.Notify.sms_sink(("[OUTPUT]</br>TaskProcessingSystem.Notify.sms_sink"))
        TaskProcessingSystem.Notify.log_sink(("[OUTPUT]</br>TaskProcessingSystem.Notify.log_sink"))
        TaskProcessingSystem.Notify.NotificationFork[TaskProcessingSystem.Notify.NotificationFork]
        TaskProcessingSystem.Notify.input --> TaskProcessingSystem.Notify.NotificationFork
        TaskProcessingSystem.Notify.NotificationFork --> TaskProcessingSystem.Notify.email_sink
        TaskProcessingSystem.Notify.NotificationFork --> TaskProcessingSystem.Notify.sms_sink
        TaskProcessingSystem.Notify.NotificationFork --> TaskProcessingSystem.Notify.log_sink
    end
    subgraph TaskProcessingSystem.ErrorHandle
        TaskProcessingSystem.ErrorHandle.input(("TaskProcessingSystem.ErrorHandle.input"))
        TaskProcessingSystem.ErrorHandle.error_log(("[OUTPUT]</br>TaskProcessingSystem.ErrorHandle.error_log"))
        TaskProcessingSystem.ErrorHandle.ErrorForward[TaskProcessingSystem.ErrorHandle.ErrorForward]
        TaskProcessingSystem.ErrorHandle.input --> TaskProcessingSystem.ErrorHandle.ErrorForward
        TaskProcessingSystem.ErrorHandle.ErrorForward --> TaskProcessingSystem.ErrorHandle.error_log
    end
    TaskProcessingSystem.completion_tracker(("[OUTPUT]</br>TaskProcessingSystem.completion_tracker"))
    TaskProcessingSystem.forward_source_to_input[TaskProcessingSystem.forward_source_to_input]
    TaskProcessingSystem.CompletionFork[TaskProcessingSystem.CompletionFork]
    TaskProcessingSystem.FailureFork[TaskProcessingSystem.FailureFork]
    TaskProcessingSystem.TaskGen.source --> TaskProcessingSystem.forward_source_to_input
    TaskProcessingSystem.forward_source_to_input --> TaskProcessingSystem.TaskProc.input
    TaskProcessingSystem.TaskProc.completed --> TaskProcessingSystem.CompletionFork
    TaskProcessingSystem.CompletionFork --> TaskProcessingSystem.Notify.input
    TaskProcessingSystem.CompletionFork --> TaskProcessingSystem.completion_tracker
    TaskProcessingSystem.TaskProc.failed --> TaskProcessingSystem.FailureFork
    TaskProcessingSystem.FailureFork --> TaskProcessingSystem.ErrorHandle.input
    TaskProcessingSystem.FailureFork --> TaskProcessingSystem.completion_tracker
```

**Finite State Machine (Septum)** - A simple idle/processing/done workflow:

```mermaid
flowchart TD
    start((start)) --> S1
    S1[IdleState]
    S1 -->|"START"| S2
    S1 -->|"QUIT"| S3
    S2[ProcessingState]
    S2 -->|"DONE"| S1
    S3[DoneState (**terminal**)]
```

## Documentation

Full documentation is available at [https://mycorrhizal.readthedocs.io](https://mycorrhizal.readthedocs.io)

## Development

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/mycorrhizal --cov-report=html
```

## Project Status

This is a 0.1.0 release. The core APIs are stable and well-tested, but some features are still in development:

* Current: Four DSLs with decorator-based syntax
* Current: Comprehensive examples and tests
* Planned: Cross-DSL interoperability layer
* Planned: Enhanced composition patterns

## License

MIT

## Author

Jeff Ciesielski
