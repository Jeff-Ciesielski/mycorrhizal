# Installation

## Requirements

- Python 3.10 or later

## Using uv (Recommended)

```bash
uv pip install mycorrhizal
```

## Using pip

```bash
pip install mycorrhizal
```

## Development Installation

For development or to run examples:

```bash
git clone https://github.com/yourusername/mycorrhizal.git
cd mycorrhizal
uv pip install -e ".[dev]"
```

## Verify Installation

```python
from mycorrhizal.enoki import enoki
from mycorrhizal.hypha import pn
from mycorrhizal.rhizomorph import bt
from mycorrhizal.spores import get_spore_sync

print("Mycorrhizal installed successfully!")
```

## Optional Dependencies

The core library has no external dependencies for basic usage. Some features may require:

- **pydantic**: Already required for blackboard models
- **asyncio**: Required for async/await support (built into Python 3.10+)

## Examples

After installation, you can run the examples:

```bash
# From the repository root
python examples/enoki_decorator_basic.py
python examples/hypha_demo.py
python examples/rhizomorph_example.py
python examples/blended_demo.py
```

For standalone examples, see the `examples/` directory in the repository.
