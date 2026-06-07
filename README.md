# bayes-factor-crowding
A system that estimates when common equity factors are becoming crowded or unstable, then uses AI to summarize plausible macro/news explanations.

## System Skeleton

This repo is organized around a leakage-free research loop:

```text
factor returns
  -> observation window
  -> Bayesian latent-regime inference
  -> posterior regime probabilities
  -> risk overlay
  -> walk-forward backtest
```

The current code includes a first concrete Bayesian regime model:

```text
portfolio_return_t | z_t = regime
  ~ Student-t(posterior predictive parameters for that regime)

z_t | z_{t-1}
  ~ sticky Markov transition matrix
```

The implemented engine is `BayesianStudentTRegimeInference`. It collapses factor returns into a portfolio return, estimates ordered normal/crowded/stress Student-t posterior predictives from the historical window, and runs an HMM filter to produce current regime probabilities.

## Layout

```text
src/bayes_factor_crowding/
  data/        load and validate factor returns
  features/    build leakage-free model inputs
  models/      latent-regime model configuration
  inference/   Bayesian Student-t filter plus mixed-tempering adapter placeholder
  signals/     convert posterior beliefs into risk multipliers
  backtest/    walk-forward loop, costs, metrics
  baselines/   simple comparison strategies
  context/     timestamp-aware market context and RAG placeholders
  experiments/ runnable demos
```

## Quickstart

```bash
python -m pip install -e ".[dev]"
python -m bayes_factor_crowding.experiments.run_demo
python -m pytest
```
