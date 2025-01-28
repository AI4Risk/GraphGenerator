# graph_metrics

A comprehensive evaluation toolkit for graph generation research, providing essential metrics to assess structural similarity, distribution distance, and generation quality between synthetic and reference graphs.

## Features

- Quantitative comparison of both static graphs and temporal graph sequences
- Support for multiple graph representations: 
  - Scipy sparse matrices (`csr_matrix`)
  - NetworkX graph objects
  - NumPy adjacency matrices
- Statistical evaluation including **mean/median** comparison methods

## Basic Usage

### Computing Graph Statistics
```python
from graph_metrics import compute_statistics
import json

# Input: Adjacency matrix (csr_matrix/NetworkX graph/NumPy array)
graph_stats = compute_statistics(your_graph)
print(json.dumps(graph_stats, indent=4))
```

### Graph Comparison Evaluation
```python
from graph_metrics import CompEvaluator

evaluator = CompEvaluator()

# Dynamic graph comparison (temporal sequences)
reference_sequence = [...]  # List of adjacency matrices
generated_sequence = [...]  # List of adjacency matrices

# Compare using mean aggregation
mean_results = evaluator.comp_graph_stats(reference_sequence, generated_sequence)
# Compare using median aggregation
median_results = evaluator.comp_graph_stats(
    reference_sequence, 
    generated_sequence,
    eval_method='med'
)

# Static graph comparison
static_reference = ...  # Single adjacency matrix
static_generated = ...  # Single adjacency matrix
static_results = evaluator.comp_graph_stats(static_reference, static_generated)
```
