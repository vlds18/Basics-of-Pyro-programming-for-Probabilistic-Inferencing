# Basics of Pyro Programming

This notebook provides a comprehensive introduction to probabilistic programming using Pyro, a flexible probabilistic programming language built on PyTorch.

## Project overview

This tutorial covers fundamental concepts from basic probabilistic models to Bayesian inference with MCMC sampling, demonstrating how to:

- Define probabilistic models in Pyro
- Perform Bayesian inference using MCMC with NUTS sampling
- Handle conditional models and observational data
- Work with multi-dimensional distributions and plates
- Visualize posterior distributions

## Key Sections

### 1. Basic Probabilistic Models
Introduction to fundamental Pyro concepts including random variable sampling and prior distributions.

### 2. Conditioned Posterior Models
Implementation of models with observational data and posterior inference.

### 3. Bayesian Coin Flip inference
Complete pipeline for estimating coin bias parameters using real observational data.

### 4. Relation Reliability Analysis
Comparative study of two scenarios:
- All positive observations conditioning
- Mixed data conditioning (70% positive)

## Core concepts

### Probabilistic Programming fundamentals
- Random variable sampling with pyro.sample()
- Prior distribution specification
- Observational data integration

### Bayesian Inference methods
- Markov Chain Monte Carlo (MCMC) sampling
- No-U-Turn Sampler (NUTS) implementation
- Posterior distribution estimation
- Predictive distribution sampling

### Advanced Pyro features
- Plate notation for independent conditional distributions
- to_event() for dependent random variables
- Multi-dimensional parameter handling

## Code structure

### Model definitions
The notebook implements several key probabilistic models:

**Basic Model:**
```python
def model():
    p = pyro.sample("p", dist.Beta(2.0, 2.0))
    x = pyro.sample("x", dist.Bernoulli(p))
    return x
```
**Conditioned Model:**
```python
def conditioned_model():
    p = pyro.sample("p", dist.Beta(2.0, 2.0))
    pyro.sample("x", dist.Bernoulli(p), obs=torch.tensor(1.0))
    return p
```
**Bayesian Coin Flip Model:**
```python
def flip_model(data=None):
    p = pyro.sample("p", dist.Beta(2.0, 2.0))
    if data is not None and len(data) > 0:
        with pyro.plate("data_plate", len(data)):
            pyro.sample("obs", dist.Bernoulli(p), obs=data)
    else:
        pyro.sample("obs", dist.Bernoulli(p))
```

**Relation Reliability Models:**
```python
def relation_model_all_ones():
    theta = pyro.sample("theta", dist.Beta(torch.ones(N_relations), torch.ones(N_relations)).to_event(1))
    with pyro.plate("data", len(idx_fixed)):
        pyro.sample("obs", dist.Bernoulli(theta[idx_fixed]), obs=torch.ones(len(idx_fixed)))
    return theta

def relation_model_mixed():
    theta = pyro.sample("theta", dist.Beta(torch.ones(N_relations), torch.ones(N_relations)).to_event(1))
    with pyro.plate("data", len(idx_fixed)):
        pyro.sample("obs", dist.Bernoulli(theta[idx_fixed]), obs=mixed_data)
    return theta
```
### Inference Setup
- MCMC with NUTS sampler configuration
- Posterior sampling with warmup steps
- Predictive distribution generation using `Predictive` class
- Comparative analysis between different data scenarios

## Implementation Details

### Sampling Configuration
- **Warmup Steps**: 200 for chain convergence
- **Num Samples**: 500-1000 for posterior estimation
- **Random Seeds**: Fixed for reproducibility using `pyro.set_rng_seed(0)`
- **Validation**: Enabled with `pyro.enable_validation(True)`

### Data Generation
- Synthetic relation reliability data with 100 samples
- Mixed dataset: 70% positive observations, 30% negative
- Randomized indexing across 3 relations
- Permuted data for unbiased sampling

### Visualization Methods
- Multi-panel figure layouts with 3 subplots
- Color-coded histogram distributions for different relations
- Comparative analysis plots between conditioning scenarios
- Posterior mean annotations and reference lines
- Proper labeling and legends for interpretability

## Results and Outputs

### Parameter Estimation
- Posterior means for coin bias parameter (p)
- Relation reliability distributions (theta)
- Uncertainty quantification through variance and distribution shapes
- Predictive samples for future observations

### Comparative Analysis
- Clear differentiation between all-ones vs mixed data conditioning
- Demonstration of Bayesian updating principles
- Visual evidence of posterior convergence
- Quantitative comparison of posterior means

## Technical Requirements

### Dependencies
- `pyro-ppl`: Probabilistic programming library
- `torch`: Backend for tensor operations
- `matplotlib`: Visualization and plotting
- Standard scientific Python stack (NumPy, etc.)

### Hardware Considerations
- CPU execution sufficient for all models
- Memory efficient sampling algorithms
- Scalable to larger datasets with adjusted parameters
- Suitable for both local and cloud execution

## Usage Instructions

1. **Initialization**: Execute environment setup cells first
2. **Model Definition**: Run model definition cells sequentially
3. **Inference**: Execute MCMC sampling with specified parameters
4. **Analysis**: Run visualization and comparison cells
5. **Customization**: Modify hyperparameters and data as needed

### Key Execution Steps:
```python
# Initialize MCMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)

# Run inference
mcmc.run(data)

# Get results
posterior_samples = mcmc.get_samples()
```
## Extensibility

The notebook structure supports:

- Additional probabilistic distributions from `pyro.distributions`
- Complex hierarchical models with multiple levels
- Custom inference algorithms and samplers
- Integration with external datasets and real-world data
- Extended visualization and analysis techniques
