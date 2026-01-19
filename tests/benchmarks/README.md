# Mycorrhizal Benchmark Suite

This directory contains throughput and performance benchmarks for the three core Mycorrhizal libraries:

- **Hypha** - Petri net DSL (tokens processed per second)
- **Rhizomorph** - Behavior tree DSL (tree ticks/executions per second)
- **Septum** - Finite state machine DSL (state transitions per second)

## Running Benchmarks

### Run All Benchmarks

```bash
# Run all benchmarks with pytest-benchmark
pytest tests/benchmarks/ -v

# Run with histogram generation (outputs to .benchmarks/)
pytest tests/benchmarks/ --benchmark-only

# Run with comparison against previous run
pytest tests/benchmarks/ --benchmark-compare
```

### Run Specific Library Benchmarks

```bash
# Hypha benchmarks only
pytest tests/benchmarks/bench_hypha.py -v

# Rhizomorph benchmarks only
pytest tests/benchmarks/bench_rhizomorph.py -v

# Septum benchmarks only
pytest tests/benchmarks/bench_septum.py -v
```

### Run Specific Benchmarks

```bash
# Run a specific benchmark
pytest tests/benchmarks/bench_hypha.py::test_simple_token_throughput -v

# Run with marker
pytest -m hypha -v
pytest -m rhizomorph -v
pytest -m septum -v
```

## Benchmark Scenarios

### Hypha (Petri Nets)

Tests measure **tokens processed per second** across different scenarios:

- **Simple processing**: Single transition processing tokens
- **Complex routing**: Multiple transitions with different paths
- **Concurrent workers**: Parallel token processing
- **Nested subnets**: Hierarchical net structures

### Rhizomorph (Behavior Trees)

Tests measure **tree ticks/executions per second** across different scenarios:

- **Simple action**: Single action node
- **Sequence composites**: Multi-node sequences
- **Selector composites**: Branching logic
- **Mixed complexity**: Realistic tree structures

### Septum (Finite State Machines)

Tests measure **state transitions per second** across different scenarios:

- **Simple transitions**: Two-state machines
- **Multi-state**: Larger state machines (5-10 states)
- **Message passing**: High-frequency message handling
- **Timeout handling**: States with timeout logic

## Interpreting Results

### Metrics

- **Operations/sec**: Primary throughput metric (higher is better)
- **Min/Max/Median**: Statistical distribution of operations
- **StdDev**: Consistency of performance (lower is more stable)

### Factors Affecting Performance

1. **Python version**: PyPy may show different characteristics than CPython
2. **System load**: Close other applications for consistent results
3. **Asyncio overhead**: All three libraries are async-first
4. **Token/data size**: Larger tokens increase processing time
5. **Tree/net complexity**: More nodes/transitions = more overhead

### Benchmark Stability

- Each benchmark runs multiple iterations automatically (pytest-benchmark)
- First iteration is a warmup (discarded from results)
- Use `--benchmark-warmup` for additional warmup iterations
- Use `--benchmark-min-rounds` to ensure sufficient sample size

## Examples

### Basic Benchmark Run

```bash
$ pytest tests/benchmarks/bench_hypha.py -v

test_simple_token_throughput [...] PASSED [ 25%]
test_complex_token_throughput [...] PASSED [ 50%]
test_concurrent_workers [...] PASSED [ 75%]
test_nested_subnets [...] PASSED [100%]

--- Benchmark Results ---
Name (time in ms)         Min       Max      Mean    StdDev
----------------------------------------------------------------
simple_token_throughput   0.1234    0.1456   0.1312   0.0089
complex_token_throughput  0.2345    0.2678   0.2456   0.0123
...
```

### Generating Performance Reports

```bash
# Generate HTML report
pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-json=output.json

# Compare against baseline
pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-compare \
  --benchmark-autosave
```

## Writing New Benchmarks

When adding new benchmarks, follow these conventions:

1. **Use pytest-benchmark decorator**:
   ```python
   def test_my_benchmark(benchmark):
       result = benchmark(my_function, arg1, arg2)
       assert result == expected
   ```

2. **Mark with appropriate library**:
   ```python
   @pytest.mark.hypha
   def test_hypha_benchmark(benchmark):
       ...
   ```

3. **Include docstring** explaining what's being measured

4. **Use appropriate operations count** for meaningful measurements

5. **Avoid I/O in benchmarked code** (focus on library overhead)

## Continuous Integration

Benchmarks can be integrated into CI for performance regression detection:

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmarks
  run: |
    pytest tests/benchmarks/ \
      --benchmark-only \
      --benchmark-json=benchmark_output.json

- name: Store benchmark results
  uses: benchmark-action/github-action-benchmark@v1
```

## Performance Optimization Tips

Based on benchmark results, consider these optimizations:

1. **Reuse tree/net instances** instead of recreating
2. **Use CycleClock for simulation** (faster than wall clock)
3. **Minimize token data size** for Hypha
4. **Use conditions instead of actions** where possible in Rhizomorph
5. **Batch messages** for high-throughput Septum FSMs
