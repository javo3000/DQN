# Neural Safety Value Network

A from-scratch NumPy implementation of the neural safety value network from ["Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports"](https://arxiv.org/abs/2504.11771) (Oh, Lidard, Hu et al., 2025).



## Overview

This repository implements the core safety value function `V(x)` that powers human-centered safety filters (HCSF) for shared autonomy systems. The safety value network learns to predict how "safe" a given state is, outputting a scalar in `[0, 1]`.

**Why NumPy?** Understanding every gradient computation before scaling to JAX for edge inference optimization.

## Architecture

```
Input State (1, 133)
       ↓
   Dense Layer 1: 133 → 256 + ReLU
       ↓
   Dense Layer 2: 256 → 256 + ReLU
       ↓
   Dense Layer 3: 256 → 1 + Sigmoid
       ↓
Safety Value V(x) ∈ [0, 1]
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Time** | 3.19 seconds |
| **Total Epochs** | 10,000 |
| **Throughput** | 3,133 epochs/sec |
| **Time per Epoch** | 0.319 ms |
| **Inference Time** | ~0.16 ms |
| **Final Error** | 0.000000 |
| **Convergence** | 100% (prediction matches target) |

## Computational Complexity

| Operation | Complexity | FLOPs |
|-----------|------------|-------|
| Forward Pass | O(d_hidden²) | ~100,000 |
| Backward Pass | O(d_hidden²) | ~165,000 |
| Per Epoch | O(d_hidden²) | ~265,000 |
| Full Training | O(epochs × d_hidden²) | ~2.65B |

**Dominant operation:** 256×256 hidden layer matrix multiplication

## Key Implementation Details

### Loss Function
Binary Cross-Entropy (BCE) with sigmoid activation. The gradient simplifies beautifully:

```python
# Instead of: d_z = (∂L/∂a) × (∂a/∂z)
# We get:
d_z = prediction - target  # One line, numerically stable
```

### Backpropagation
Hand-implemented with full gradient tracking:
- Weight gradients: `grad_W = input.T @ delta`
- Bias gradients: `grad_b = sum(delta, axis=0)`
- Error propagation: `delta_prev = delta @ W.T * activation_derivative(z)`

### Training Loop
```python
for epoch in range(epochs):
    # Forward pass
    prediction = model.forward_pass(state, target)
    
    # Backward pass (called automatically in forward_pass)
    # Updates weights using: W -= learning_rate * grad_W
```

## Usage

```bash
python Human-centered-safety-filter.py
```

### Output
```
Epoch: 1000, Prediction: 0.890000, Error: -0.000000, Grad W3: 0.000000...
Epoch: 2000, Prediction: 0.890000, Error: -0.000000, Grad W3: 0.000000...
...
Epoch: 10000, Prediction: 0.890000, Error: 0.000000, Grad W3: 0.000000...

Training completed in 3.191 seconds
   Throughput: 3133 epochs/sec
   Time per epoch: 0.319 ms

Final Results:
  Target: 0.89
  Prediction: 0.890000
  Error: 0.000000

Learning curve saved to: learning_curve.png
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.01 |
| Epochs | 10,000 |
| Hidden Dimensions | 256 |
| Input Dimensions | 133 |
| Output | Scalar ∈ [0, 1] |
| Weight Init | Gaussian × 0.01 |
| Bias Init | Zeros |

## Files

```
Code walkthroughs/
├── Human-centered-safety-filter.py   # Main implementation
├── learning_curve.png                 # Training visualization
├── social_media_content.md           # Twitter/LinkedIn content
└── README.md                          # This file
```

## Next Steps

1. **JAX Migration** - Port to JAX for XLA compilation and GPU acceleration
2. **Q-CBF Implementation** - Add the control barrier function intervention layer
3. **Edge Optimization** - Benchmark on edge hardware for real-time deployment
4. **Batched Training** - Extend to multiple state samples per epoch

## Paper Reference

```bibtex
@article{oh2025safety,
  title={Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports},
  author={Oh, Donggeon David and Lidard, Justin and Hu, Haimin and Sinhmar, Himani and 
          Lazarski, Elle and Gopinath, Deepak and Sumner, Emily S. and DeCastro, Jonathan A. and 
          Rosman, Guy and Leonard, Naomi Ehrich and Fisac, Jaime Fern{\'a}ndez},
  journal={arXiv preprint arXiv:2504.11771},
  year={2025}
}
```

## Key Contributions from the Paper

- **First fully model-free CBF safety filter** - No dynamics model required
- **Q-CBF (State-Action Control Barrier Function)** - Works with black-box systems
- **Human-centered interventions** - Smooth, minimal corrections that preserve agency
- **Validated with 83 human participants** - Statistically significant improvements in safety and satisfaction

## License

MIT License - See LICENSE file for details.

---

*Building safety filters that work with humans, not against them.*
