# Universal-Min-Hash-Based-RAG
A universal MinHash-based library for computing similarity between different data structures including sets, sequences, trees, and weighted sets.

## Overview

SimilarityHashing provides a unified interface for applying locality-sensitive hashing (LSH) techniques to various data structures. It uses MinHash algorithms to approximate Jaccard similarity efficiently, making it suitable for large-scale similarity computation and nearest neighbor search.

## Features

- **Multiple Data Structure Support**: Handle sets, sequences, trees, and weighted sets
- **Efficient Similarity Computation**: Fast approximate Jaccard similarity using MinHash
- **Configurable Hash Size**: Adjustable hash signatures for speed vs. accuracy trade-offs
- **Tree Structure Analysis**: Novel approach to tree similarity using branch decomposition
- **Sequence Order Sensitivity**: Captures positional relationships in ordered data
- **Weighted Set Support**: Handles frequency-based similarity for weighted data

## Installation

```bash
pip install datasketch numpy
```

**Note**: The code uses PyPy for optimal performance. Install PyPy for best results.

## Quick Start

```python
from similarity_hashing import SimilarityHashing

# Initialize for different data structures
set_hasher = SimilarityHashing('set', size=4096)
sequence_hasher = SimilarityHashing('sequence', size=4096)
tree_hasher = SimilarityHashing('tree', size=4096)
weighted_hasher = SimilarityHashing('weighted_set', size=64, weighted_option='ports')

# Hash your data
set_hash1 = set_hasher.hash_data({1, 2, 3, 4})
set_hash2 = set_hasher.hash_data({1, 2, 5, 6})

# Compute similarity
similarity = set_hash1.jaccard(set_hash2)
print(f"Jaccard similarity: {similarity}")
```

## Supported Data Structures

### 1. Sets
Standard set similarity using MinHash.

```python
hasher = SimilarityHashing('set', size=4096)
hash1 = hasher.hash_data({1, 2, 3, 4})
hash2 = hasher.hash_data({3, 4, 5, 6})
similarity = hash1.jaccard(hash2)  # ~0.33 (2 common / 6 total)
```

### 2. Sequences
Order-sensitive similarity using pairwise combinations. Best for longer sequences.

```python
hasher = SimilarityHashing('sequence', size=4096)
hash1 = hasher.hash_data((1, 2, 3, 4))
hash2 = hasher.hash_data((1, 3, 2, 4))  # Different order
similarity = hash1.jaccard(hash2)
```

**How it works**: Converts sequences to sets of ordered pairs using `combinations(sequence, 2)`, capturing positional relationships.

### 3. Trees
Tree similarity based on structural decomposition.

```python
hasher = SimilarityHashing('tree', size=4096)
tree1 = {1: {2: {3: 4}, 5: {6: 7}}}
tree2 = {1: {2: {3: 4}, 5: {8: 9}}}
hash1 = hasher.hash_data(tree1)
hash2 = hasher.hash_data(tree2)
similarity = hash1.jaccard(hash2)
```

**How it works**: 
1. Flattens tree into all root-to-leaf paths
2. Creates pairwise combinations within each path
3. Applies set hashing to the resulting pairs

### 4. Weighted Sets
Frequency-aware similarity for weighted data.

```python
hasher = SimilarityHashing('weighted_set', size=64, weighted_option='ports')
hash1 = hasher.hash_data({'1': 10, '2': 20, '3': 5})
hash2 = hasher.hash_data({'1': 15, '2': 18, '4': 7})
similarity = hash1.jaccard(hash2)
```

**Weighted options**:
- `'ports'`: 65536 dimensions (for port numbers)
- `'english'`: 100 dimensions (for small vocabularies)
- `'sfc'`: 10 dimensions (for small feature counts)

## API Reference

### Class: SimilarityHashing

#### Constructor
```python
SimilarityHashing(structure, size, weighted_option='None')
```

**Parameters**:
- `structure` (str): Data structure type - `'set'`, `'sequence'`, `'tree'`, or `'weighted_set'`
- `size` (int): Number of hash functions (higher = more accurate, slower)
- `weighted_option` (str): Required for weighted_set - `'ports'`, `'english'`, or `'sfc'`

#### Methods

##### `hash_data(data)`
Converts input data to MinHash signature.

**Parameters**:
- `data`: Input data in the appropriate format for the chosen structure

**Returns**: MinHash object with `.jaccard()` method for similarity computation

## Performance Considerations

### Hash Size Trade-offs
- **Larger hash size** (e.g., 4096): Higher accuracy, more memory, slower computation
- **Smaller hash size** (e.g., 64): Lower accuracy, less memory, faster computation

### Data Structure Specific Notes

- **Sets**: Most efficient, direct MinHash application
- **Sequences**: Quadratic complexity in sequence length due to pairwise combinations
- **Trees**: Complexity depends on tree depth and branching factor
- **Weighted Sets**: Most memory-intensive due to vector operations

## Example Use Cases

### Document Similarity
```python
# Compare documents as sets of words
doc_hasher = SimilarityHashing('set', size=2048)
doc1_words = set("the quick brown fox jumps".split())
doc2_words = set("the fast brown fox leaps".split())

hash1 = doc_hasher.hash_data(doc1_words)
hash2 = doc_hasher.hash_data(doc2_words)
similarity = hash1.jaccard(hash2)
```

### Time Series Similarity
```python
# Compare sequences while preserving order
seq_hasher = SimilarityHashing('sequence', size=4096)
series1 = (1, 3, 5, 7, 9)
series2 = (1, 3, 6, 7, 9)

hash1 = seq_hasher.hash_data(series1)
hash2 = seq_hasher.hash_data(series2)
similarity = hash1.jaccard(hash2)
```

### Hierarchical Data Comparison
```python
# Compare organizational structures
tree_hasher = SimilarityHashing('tree', size=4096)
org1 = {'CEO': {'Engineering': {'Frontend': 'Team1', 'Backend': 'Team2'}, 'Sales': {'Region1': 'Rep1'}}}
org2 = {'CEO': {'Engineering': {'Frontend': 'Team1', 'ML': 'Team3'}, 'Marketing': {'Digital': 'Specialist1'}}}

hash1 = tree_hasher.hash_data(org1)
hash2 = tree_hasher.hash_data(org2)
similarity = hash1.jaccard(hash2)
```

## Implementation Details

### Tree Flattening Algorithm
The `flatten_tree()` function converts nested dictionaries into sequences of key-value paths:

```python
# Input: {1: {2: {3: 4}, 5: 6}}
# Output: [(1, 2, 3, 4), (1, 5, 6)]
```

### Sequence Hashing Strategy
Uses `itertools.combinations(data, 2)` to capture order relationships while maintaining set-based similarity computation.

### Weighted Set Implementation
Converts weighted dictionaries to sparse vectors using NumPy, then applies WeightedMinHash for similarity-preserving dimensionality reduction.

## Dependencies

- `datasketch`: MinHash and WeightedMinHashGenerator
- `numpy`: Vector operations for weighted sets
- `itertools`: Combination generation for sequences
