# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mycorrhizal is a Python library for building safe, structured, concurrent, event-driven systems. It provides three DSL (Domain-Specific Language) systems that can be used independently or combined:

1. **Hypha** - Colored Object-Centric Petri Nets for workflow modeling
2. **Rhizomorph** - Asyncio-friendly Behavior Trees for decision-making and control logic
3. **Septum** - Decorator-based Finite State Machines with asyncio support

All three systems use decorator-based syntax and share a common infrastructure for time management and state (blackboard) handling.

## Integration with other systems

The spores system is being co-developed with acorn/moria (soon to be renamed acorn-core), which is a real-time process mining framework.

The Acorn record format: /home/jeff/workspace/kudzu/acorn/moria/docs/RECORD_FORMAT.md doc should be consulted for details on the expected json structure for event logs.

## Development Commands

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m slow              # Slow tests only

# Run with coverage reports
pytest --cov=src/mycorrhizal --cov-report=html
```

### Package Management
- Uses UV as the package manager (see `uv.lock`)
- Build system: Hatchling
- Python version: 3.10+

## Coding Standards

### No Emojis
- **DO NOT use emojis** in user-facing code, comments, documentation, or commit messages
- This includes variable names, UI text, log messages, error messages, and any output
- Use plain text instead - e.g., "start" instead of "▶️", "error" instead of "❌", "initial" instead of "⚡"
- Don't make claims about 'production ready' in examples or documentation
- NEVER use the phrase 'key points', never use an em-dash.

## Architecture

### Core Design Principles

**1. Explicit Data Flow > Shared Memory**

Petri net transitions should move tokens between places, not mutate shared state. Data should flow through the net structure (InputPlace → Transition → OutputPlace). Side effects in transitions break analyzability and reasoning about the system.

**2. Composition > Concurrent Cooperation**

Primary use case: embedding one DSL inside another (e.g., behavior tree inside Petri net transition). Secondary use case: concurrent systems with coordinated access to shared state. Interfaces should enable safe composition, not just safe shared memory access.

**3. Maintain Analyzability**

Systems should be statically analyzable from their structure. Data flow should be explicit in the DSL structure, not hidden in side effects. Interfaces constrain access but don't justify breaking data flow principles.

**4. Petri Net Semantics**

Transitions consume tokens from input places and produce tokens for output places. Transitions are NOT general-purpose async functions with side effects. Blackboard access in transitions should be read-only (configuration, context). State mutations happen through token flow, not blackboard mutations.

### Example: Wrong vs Right Patterns

**WRONG - Mutating blackboard in Petri transition:**
```python
@builder.transition()
async def process_task(consumed, bb: TaskInterface, timebase):
    bb.tasks_completed += 1  # Side effect!
    bb.current_velocity = 0.5  # Breaks Petri net semantics!
    yield {output: result}
```

This pattern:
- Breaks analyzability (can't see data flow from net structure)
- Violates Petri net semantics (transitions shouldn't mutate state)
- Makes the system harder to reason about
- Defeats the purpose of having structured DSLs

**RIGHT - Embed behavior tree in transition, data flows through tokens:**
```python
@builder.transition()
async def decide(consumed, bb: ConfigInterface, timebase):
    """
    Transition that executes a behavior tree.
    Data flows: input token → BT decision → output token
    """
    token_data = consumed[0]

    # Execute behavior tree (composition!)
    bt_result = await execute_bt(
        tree=DecisionTree,
        input_data=token_data
    )

    # Route to output based on BT result
    if bt_result.status == Status.SUCCESS:
        yield {success_place: bt_result.data}
    else:
        yield {failure_place: token_data}
```

This pattern:
- Embeds one DSL inside another (composition!)
- Data flows explicitly through the net
- Blackboard access is read-only (config only)
- System is analyzable from structure

### Core Systems

#### Septum (Finite State Machines)
Location: `src/mycorrhizal/septum/core.py`

Septum implements a structured Finite State Machine framework with asyncio support. Key features:

- **Decorator-based states**: States are defined using function decorators (@septum.state)
- **Declarative transitions**: Enum-based transitions enable static analysis and validation
- **String-based state resolution**: Breaks circular import dependencies by using fully-qualified state names
- **Asyncio-native**: Built-in timeout support and message passing via priority queues
- **Push/Pop stack**: Enables hierarchical state machines with stack-based navigation
- **Comprehensive validation**: Construction-time validation of all reachable states and transitions

Key API:
```python
from mycorrhizal.septum.core import septum, StateMachine, StateConfiguration, LabeledTransition, Push, Pop

@septum.state(config=StateConfiguration(timeout=5.0))
def MyState():
    class Events(Enum):
        DONE = auto()
        RETRY = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Main state logic
        return Events.DONE

    @septum.on_enter
    async def on_enter(ctx: SharedContext):
        # Called when entering state
        pass

    @septum.on_timeout
    async def on_timeout(ctx: SharedContext):
        # Handle timeout
        return Events.ERROR

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.DONE, NextState),
            LabeledTransition(Events.RETRY, Retry),
        ]

# Create and run FSM
fsm = StateMachine(initial_state=MyState)
await fsm.initialize()
await fsm.run()
```

Transition types include:
- State references - Direct transition to another state
- `Again` - Re-execute current state immediately
- `Unhandled` - Wait for next message/event
- `Retry` - Re-enter state with retry counter
- `Restart` - Reset retry counter and wait for message
- `Repeat` - Re-enter state from on_enter
- `Push(state1, state2, ...)` - Push states onto stack
- `Pop` - Pop and return to previous state

#### Hypha (Petri Nets)
Location: `src/mycorrhizal/hypha/core/`

Hypha implements a decorator-based Petri Net DSL with:
- **Places** (containers for tokens): Can be queues, sets, or boolean flags
- **Transitions** (processing steps): Consume tokens from input places and produce tokens for output places
- **Arcs** (connections): Connect places to transitions and vice versa
- **Subnets** (hierarchical composition): Enable modular net construction

Key API:
```python
from mycorrhizal.hypha.core import pn, Runner as PNRunner

@pn.net
def MyNet(builder):
    my_place = builder.place("my_place")

    @builder.transition()
    async def my_transition(consumed, bb, timebase):
        yield {my_place: consumed[0]}
```

#### Rhizomorph (Behavior Trees)
Location: `src/mycorrhizal/rhizomorph/core.py`

Rhizomorph implements an async-first Behavior Tree DSL with:
- **Actions** (leaf nodes): Perform operations and return Status
- **Conditions** (leaf nodes): Return boolean or Status
- **Composites** (internal nodes): Control flow (sequence, selector, parallel, etc.)
- **Subtrees** (cross-tree composition): Enable modular tree construction with `bt.subtree()`

Key API:
```python
from mycorrhizal.rhizomorph.core import bt, Runner as BTRunner, Status

@bt.tree
def MyBT():
    @bt.action
    async def my_action(bb: Blackboard) -> Status: ...

    @bt.condition
    def my_condition(bb: Blackboard) -> bool: ...

    @bt.root
    @bt.sequence
    def root(N):
        yield N.my_condition
        yield N.my_action
```

### Shared Infrastructure

#### Blackboard Pattern
Both systems use a shared blackboard (Pydantic BaseModel) for state management:
- Holds application state
- Passed to all nodes/transitions
- Enables integration between Petri nets and behavior trees

#### Timebase
Location: `src/mycorrhizal/common/timebase.py`

Abstract time abstraction with multiple implementations:
- `WallClock` - Real wall time
- `UTCClock` - UTC wall time
- `MonotonicClock` - Monotonic system time
- `CycleClock` - Stepped time for simulation/testing
- `DictatedClock` - Programmatically controlled time

Key insight: Both Hypha and Rhizomorph can share a timebase for coordinated timing.

### Integration Pattern

The two systems can work together (see `examples/blended_demo.py`):
1. Hypha Petri net generates task tokens
2. A transition passes tokens to a behavior tree runner via shared blackboard
3. Behavior tree processes tokens over multiple ticks
4. Processed tokens returned to Petri net for further workflow

## Key Design Patterns

### Class-Based DSL (Septum)
Septum uses a class-based pattern with metaclasses:
- States inherit from `State` base class
- Methods are automatically converted to classmethods via metaclass
- Transitions defined as class methods returning enums
- Enables static analysis and validation at construction time

### Decorator-Based DSL
Hypha and Rhizomorph use declarative decorator syntax for defining structure:
- `@pn.net`, `@pn.place`, `@pn.transition` for Petri nets
- `@bt.tree`, `@bt.action`, `@bt.condition`, `@bt.root` for behavior trees

### Mermaid Diagram Export
All three systems support Mermaid diagram generation for documentation and debugging:
- `my_net.to_mermaid()` - Export Petri net structure
- `my_tree.to_mermaid()` - Export behavior tree structure
- `fsm.generate_mermaid_flowchart()` - Export FSM with push/pop analysis

### Owner-Aware Composition (Rhizomorph)
Behavior trees use `N.member` syntax for type-safe references:
- Enables cross-module composition
- Supports context-aware expansion with `bt.subtree()`
- No magic strings - all references are type-safe

### Async-First Design
- All nodes/transitions can be async functions
- Built on asyncio throughout
- Support for synchronous and async functions in same tree/net

### Type Safety
- Heavy use of Pydantic models for blackboard state
- Type hints throughout
- Runtime type checking via typeguard

## Examples

- `examples/septum/septum_decorator_basic.py` - Basic Septum FSM with decorator API
- `examples/septum/septum_decorator_timeout.py` - Septum FSM with timeout and retry handling
- `examples/blended_demo.py` - Shows Hypha + Rhizomorph integration
- `examples/hypha_demo.py` - Petri net usage
- `examples/rhizomorph_example.py` - Behavior tree usage

Run examples with:
```bash
python examples/septum/septum_decorator_basic.py
```

## Testing Notes

- Tests use pytest with asyncio_mode = "auto"
- Test timeout: 30 seconds per test
- Coverage reports generated in HTML and terminal formats
- Tests are organized by markers: unit, integration, slow, performance
- Always run ruff and pyright after making changes

## Development Notes

- All DSLs support synchronous and async functions
- Mermaid diagram export available for all three systems (useful for debugging/documentation)
- `src/mycorrhizal/hypha/core_old.py` is legacy code - do not modify

## Mycelium - Unified Orchestration Layer

Mycelium is the unified orchestration layer that allows seamless integration and nesting of FSMs, BTs, and PNs. It provides wrapper/decorator systems that extend the core APIs without modifying them.

### Key Concepts

**Leaf Node Pattern**: Vanilla Septum, Rhizomorph, and Hypha modules serve as "leaf nodes" that get integrated into Mycelium. You can develop libraries of interesting FSMs/BTs/PNs (like a model zoo), then compose them in Mycelium.

**Arbitrary Nesting**: Once inside Mycelium, you can seamlessly nest systems in any combination:
- FSM-in-BT: FSM states controlled by behavior tree actions
- BT-in-PN: Behavior trees running in Petri net transitions
- BT-in-FSM: BT subtrees as FSM states
- Deeper combinations like PN-in-FSM-in-BT, etc.

**Seamless Visualizations**: A critical feature - Mycelium generates unified Mermaid diagrams showing all nested systems together:
- Petri net with BT/FSM embedded in transitions
- BT with FSM actions as subgraphs
- Single diagram showing complete integrated system

**Spores Integration**: Event logging works across all Mycelium compositions - record FSMs, BTs, PNs, or any combination seamlessly.

### Module Structure

- `src/mycorrhizal/mycelium/` - Core Mycelium implementation
  - `core.py` - Tree decorator and composition system
  - `hypha_bridge.py` - BT-in-PN wrapper (does NOT modify hypha core)
  - `runner.py` - TreeRunner for executing composed systems
  - `instance.py` - TreeInstance for execution context
- `examples/mycelium/` - Mycelium integration examples
  - `ci_cd_orchestrator.py` - FSM+BT integration
  - `robot_controller.py` - BT with FSM actions
  - `job_queue_processor.py` - BT-in-PN integration

### Long-term Vision / Mycelium Strategy

**CRITICAL** The next bit is crucial for guiding development priorities w/r/t Mycelium vs core modules.
**V1 Goal**: Mycelium becomes the canonical way to design FSMs/BTs/PNs. The core modules (Septum, Rhizomorph, Hypha) will exist primarily as runtimes, while most users develop Mycelium implementations to enable seamless interoperability.

This means:
- Core modules stay clean and focused on runtime execution (and visualization for their own systems)
    - THIS IS CRITICAL - DO NOT ADD COMPOSITION LOGIC TO CORE MODULES
- Mycelium provides all composition, integrated-visualization, and integration features
- Users can arbitrarily compose systems without hitting API limitations
- Documentation should guide users to start with Mycelium for most use cases but
  be clear that core modules can be used standalone if desired and that it is
  perfectly OK to use a single paradigm (e.g., just FSMs) if that meets their
  needs via mycelium

**Note**: This vision context is for Claude's understanding during development. It should NOT appear in user-facing documentation or module READMEs, which should remain factual and focused on current capabilities.
- Always make sure that tests pass and examples run before marking a feature as complete

## Important Notes

**CRITICAL**: Always use UV for python operations. Running scripts, running tests, installing packages, etc. should be done via `uv run python ...` or `uv run pytest ...`. This ensures the correct virtual environment and dependencies are used.