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

## Model Comparison insights

The notebook demonstrates how different data scenarios affect posterior estimates:

- **All Positive Data**: Posterior reliabilities converge toward 1.0
- **Mixed Data (70% positive)**: Posterior reliabilities center around true proportion

## Visualization features

Comprehensive plotting of:
- Posterior parameter distributions
- Comparative analysis across conditioning scenarios
- Histogram representations of sampled parameters

## Applications

This foundation enables:
- Bayesian A/B testing frameworks
- System reliability estimation
- Probabilistic decision support systems
- Machine learning uncertainty quantification

## Educational value

Serves as prerequisite knowledge for advanced topics including:
- Variational Inference methods
- Deep probabilistic modeling
- Bayesian neural networks
- Hierarchical model structures
- Probabilistic time series analysis

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
