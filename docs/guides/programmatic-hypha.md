# Programmatic Petri Net Construction

This guide explains how to build Petri nets programmatically using the NetBuilder API, as an alternative to the decorator-based DSL.

## Overview

The Hypha DSL provides two ways to build Petri nets:

1. **Decorator DSL** (`@pn.net`) - Declarative, decorator-based syntax
2. **Programmatic API** (`NetBuilder`) - Imperative, runtime construction

Both approaches are fully capable and produce equivalent nets. The choice depends on your use case.

## When to Use Programmatic Building

**Use the programmatic API when:**
- Building nets from configuration files (JSON, YAML, etc.)
- Constructing workflows dynamically at runtime
- Need conditional net structures based on runtime data
- Generating nets programmatically from external definitions
- Building workflow engines or visual editors

**Use the decorator DSL when:**
- Net structure is static and known at development time
- You prefer declarative, readable syntax
- Building application-specific workflows

## NetBuilder API Reference

### Basic Usage

```python
from mycorrhizal.hypha.core.builder import NetBuilder
from mycorrhizal.hypha.core import PlaceType, Runner

# Create a builder
builder = NetBuilder("MyNet")

# Create places
input_place = builder.place("input", type=PlaceType.QUEUE)
output_place = builder.place("output", type=PlaceType.BAG)

# Create transition
async def my_transition(consumed, bb, timebase):
    for token in consumed:
        yield {output_place: f"processed_{token}"}

transition = builder.transition()(my_transition)

# Wire together
builder.arc(input_place, transition).arc(output_place)

# Create wrapper for Runner
def net_func():
    pass

net_func._spec = builder.spec
net_func.to_mermaid = lambda: builder.spec.to_mermaid()

# Use with Runner
runner = Runner(net_func, blackboard)
```

### Key Methods

#### Creating Places

```python
# BAG place (multiset, allows duplicates)
bag = builder.place("my_bag", type=PlaceType.BAG)

# QUEUE place (FIFO queue)
queue = builder.place("my_queue", type=PlaceType.QUEUE)
```

#### IO Input Places

Decorator for IO input places (token sources):

```python
@builder.io_input_place()
async def source(bb, timebase):
    """Generator that yields tokens"""
    for i in range(10):
        yield f"token_{i}"
```

#### IO Output Places

Decorator for IO output places (token sinks):

```python
@builder.io_output_place()
async def sink(bb, timebase, tokens):
    """Consumes tokens from the place"""
    for token in tokens:
        print(f"Output: {token}")
```

#### Creating Transitions

```python
@builder.transition()
async def my_transition(consumed, bb, timebase):
    """Process tokens from input places"""
    for token in consumed:
        # Process token
        yield {output_place: processed_token}
```

#### Connecting with Arcs

The `arc()` method connects places and transitions. Returns `ArcChain` for chaining:

```python
# Single arc
builder.arc(input_place, transition)

# Chained arcs
builder.arc(source, trans1).arc(place1).arc(trans2).arc(output)
```

#### Convenience Methods

**Forward** - Create a pass-through transition:

```python
# Creates a transition that moves tokens from input to output
builder.forward(input_place, output_place, name="pass_through")
```

**Fork** - Split tokens to multiple output places:

```python
builder.fork(input_place, [output1, output2, output3])
```

**Join** - Merge tokens from multiple input places:

```python
builder.join([input1, input2, input3], output_place)
```

#### Subnets

Instantiate a subnet within this net:

```python
from mycorrhizal.hypha.core import pn

@pn.net
def MySubnet(builder: NetBuilder):
    # Subnet definition
    ...

# Instantiate subnet in parent net
subnet_ref = builder.subnet(MySubnet, "instance1")
```

## Examples

### Example 1: Simple Linear Workflow

```python
from mycorrhizal.hypha.core.builder import NetBuilder
from mycorrhizal.hypha.core import PlaceType, Runner

def build_linear_workflow():
    """Build a simple linear processing pipeline"""
    builder = NetBuilder("LinearWorkflow")

    # Create places
    input_p = builder.place("input", type=PlaceType.QUEUE)
    stage1 = builder.place("stage1", type=PlaceType.QUEUE)
    stage2 = builder.place("stage2", type=PlaceType.QUEUE)
    output_p = builder.place("output", type=PlaceType.BAG)

    # Create transitions
    @builder.transition()
    async def process_stage1(consumed, bb, timebase):
        for token in consumed:
            yield {stage1: f"stage1_{token}"}

    @builder.transition()
    async def process_stage2(consumed, bb, timebase):
        for token in consumed:
            yield {stage2: f"stage2_{token}"}

    @builder.transition()
    async def finalize(consumed, bb, timebase):
        for token in consumed:
            yield {output_p: f"final_{token}"}

    # Wire together
    builder.arc(input_p, process_stage1).arc(stage1)
    builder.arc(stage1, process_stage2).arc(stage2)
    builder.arc(stage2, finalize).arc(output_p)

    # Create wrapper
    def net_func():
        pass

    net_func._spec = builder.spec
    net_func.to_mermaid = lambda: builder.spec.to_mermaid()

    return net_func
```

### Example 2: Dynamic Net from Configuration

```python
def build_net_from_config(config):
    """Build a Petri net from a configuration dictionary"""

    builder = NetBuilder(config["name"])

    # Build places
    places = {}
    for place_config in config["places"]:
        place_type = PlaceType.QUEUE if place_config["type"] == "queue" else PlaceType.BAG
        places[place_config["name"]] = builder.place(
            place_config["name"],
            type=place_type
        )

    # Build transitions
    transitions = {}
    for trans_config in config["transitions"]:
        # Handler functions should be registered somehow
        handler = get_handler(trans_config["handler"])
        trans = builder.transition()(handler)
        transitions[trans_config["name"]] = trans

    # Build arcs
    for trans_config in config["transitions"]:
        trans = transitions[trans_config["name"]]

        for arc_config in trans_config["arcs"]:
            source_name, target_name = arc_config

            source = places.get(source_name) or transitions.get(source_name)
            target = places.get(target_name) or transitions.get(target_name)

            builder.arc(source, target)

    # Return wrapper
    def net_func():
        pass

    net_func._spec = builder.spec
    net_func.to_mermaid = lambda: builder.spec.to_mermaid()

    return net_func

# Usage:
config = {
    "name": "DynamicNet",
    "places": [
        {"name": "input", "type": "queue"},
        {"name": "output", "type": "bag"}
    ],
    "transitions": [
        {
            "name": "process",
            "handler": process_handler,
            "arcs": [("input", "process"), ("process", "output")]
        }
    ]
}

net = build_net_from_config(config)
```

### Example 3: Conditional Net Construction

```python
def build_adaptive_workflow(use_validation: bool):
    """Build different workflow structures based on runtime conditions"""

    builder = NetBuilder("AdaptiveWorkflow")

    input_p = builder.place("input", type=PlaceType.QUEUE)
    output_p = builder.place("output", type=PlaceType.BAG)

    if use_validation:
        # Path with validation step
        validation = builder.place("validation", type=PlaceType.BAG)

        @builder.transition()
        async def validate(consumed, bb, timebase):
            for token in consumed:
                if is_valid(token):  # Some validation logic
                    yield {validation: token}

        @builder.transition()
        async def process(consumed, bb, timebase):
            for token in consumed:
                result = process_token(token)
                yield {output_p: result}

        builder.arc(input_p, validate).arc(validation)
        builder.arc(validation, process).arc(output_p)
    else:
        # Direct path (no validation)
        @builder.transition()
        async def process_direct(consumed, bb, timebase):
            for token in consumed:
                result = process_token(token)
                yield {output_p: result}

        builder.arc(input_p, process_direct).arc(output_p)

    # Create wrapper
    def net_func():
        pass

    net_func._spec = builder.spec
    net_func.to_mermaid = lambda: builder.spec.to_mermaid()

    return net_func
```

### Example 4: Using Convenience Methods

```python
def build_parallel_processing():
    """Build a parallel processing network"""

    builder = NetBuilder("ParallelProcessing")

    # Create places
    input_p = builder.place("input", type=PlaceType.QUEUE)
    worker1_p = builder.place("worker1", type=PlaceType.QUEUE)
    worker2_p = builder.place("worker2", type=PlaceType.QUEUE)
    worker3_p = builder.place("worker3", type=PlaceType.QUEUE)
    output_p = builder.place("output", type=PlaceType.BAG)

    # Fork: distribute work to 3 workers
    builder.fork(input_p, [worker1_p, worker2_p, worker3_p])

    # Add processing transitions for each worker
    for i, worker_p in enumerate([worker1_p, worker2_p, worker3_p], 1):
        @builder.transition()
        async def work(consumed, bb, timebase):
            for token in consumed:
                result = process_worker(token, worker_id=i)
                yield {output_p: result}

        builder.arc(worker_p, work).arc(output_p)

    # Create wrapper
    def net_func():
        pass

    net_func._spec = builder.spec
    net_func.to_mermaid = lambda: builder.spec.to_mermaid()

    return net_func
```

## Comparison: Decorator vs Programmatic

### Decorator Approach

```python
@pn.net
def MyNet(builder: NetBuilder):
    @builder.io_input_place()
    async def source(bb, timebase):
        yield "token1"
        yield "token2"

    output = builder.place("output", type=PlaceType.BAG)

    @builder.transition()
    async def process(consumed, bb, timebase):
        for token in consumed:
            yield {output: token}

    builder.arc(source, process).arc(output)
```

**Pros:**
- Clean, declarative syntax
- Easier to read and maintain
- Less boilerplate

**Cons:**
- Structure fixed at definition time
- Cannot be modified at runtime
- Not suitable for dynamic workflows

### Programmatic Approach

```python
def build_my_net():
    builder = NetBuilder("MyNet")

    async def source(bb, timebase):
        yield "token1"
        yield "token2"

    input_p = builder.io_input_place()(source)
    output_p = builder.place("output", type=PlaceType.BAG)

    async def process(consumed, bb, timebase):
        for token in consumed:
            yield {output_p: token}

    trans = builder.transition()(process)

    builder.arc(input_p, trans).arc(output_p)

    def net_func():
        pass

    net_func._spec = builder.spec
    net_func.to_mermaid = lambda: builder.spec.to_mermaid()

    return net_func
```

**Pros:**
- Full control over net structure
- Can build from configuration/data
- Runtime flexibility
- Suitable for workflow engines

**Cons:**
- More verbose
- Slightly more complex
- Need to create wrapper function

## Advanced Patterns

### Building Hierarchical Nets with Subnets

```python
from mycorrhizal.hypha.core import pn

# Define a reusable subnet
@pn.net
def ProcessorSubnet(builder: NetBuilder):
    input_p = builder.place("input", type=PlaceType.QUEUE)
    output_p = builder.place("output", type=PlaceType.BAG)

    @builder.transition()
    async def process(consumed, bb, timebase):
        for token in consumed:
            yield {output_p: f"processed_{token}"}

    builder.arc(input_p, process).arc(output_p)

# Use subnet in parent net
def build_hierarchical_net():
    builder = NetBuilder("HierarchicalNet")

    main_input = builder.place("main_input", type=PlaceType.QUEUE)
    main_output = builder.place("main_output", type=PlaceType.BAG)

    # Instantiate subnet
    processor1 = builder.subnet(ProcessorSubnet, "processor1")
    processor2 = builder.subnet(ProcessorSubnet, "processor2")

    # Connect to subnets
    # Note: subnet places are accessed via SubnetRef
    # See the full example for details

    # Create wrapper
    def net_func():
        pass

    net_func._spec = builder.spec
    net_func.to_mermaid = lambda: builder.spec.to_mermaid()

    return net_func
```

### Arc Chaining with ArcChain

The `arc()` method returns an `ArcChain` object for fluent chaining:

```python
builder.arc(place1, trans1)  # Returns ArcChain pointing to trans1
    .arc(place2)              # Chain: trans1 -> place2
    .arc(trans2)              # Chain: place2 -> trans2
    .arc(place3);             # Chain: trans2 -> place3
```

This is equivalent to:

```python
builder.arc(place1, trans1)
builder.arc(trans1, place2)
builder.arc(place2, trans2)
builder.arc(trans2, place3)
```

## Best Practices

### 1. Name Your Places and Transitions Clearly

```python
# Good
input_queue = builder.place("input_queue", type=PlaceType.QUEUE)
processed_data = builder.place("processed_data", type=PlaceType.BAG)

# Avoid
p1 = builder.place("p1", type=PlaceType.QUEUE)
p2 = builder.place("p2", type=PlaceType.BAG)
```

### 2. Use Consistent Conventions for Wrapper Functions

```python
def create_wrapper(spec):
    def net_func():
        pass
    net_func._spec = spec
    net_func.to_mermaid = lambda: spec.to_mermaid()
    return net_func
```

### 3. Document Configuration-Based Builders

```python
def build_from_yaml(yaml_path: str):
    """
    Build a Petri net from a YAML configuration file.

    YAML structure:
        name: MyNet
        places:
          - name: input
            type: queue
          - name: output
            type: bag
        transitions:
          - name: process
            handler: my_module.process_handler
            arcs:
            - [input, process]
            - [process, output]
    """
    # Implementation...
```

### 4. Validate Configurations Before Building

```python
def validate_config(config):
    """Validate net configuration before building"""
    required_keys = {"name", "places", "transitions"}
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    # Validate place references
    place_names = {p["name"] for p in config["places"]}
    for trans in config["transitions"]:
        for arc in trans["arcs"]:
            if arc[0] not in place_names or arc[1] not in place_names:
                # Assuming transition names are also in places
                pass
```

## See Also

- [Hypha Overview](../hypha/) - Decorator-based DSL
- [Composition Patterns](composition.md) - Subnet composition patterns
- [API Reference](../api/hypha.md) - Complete API documentation
