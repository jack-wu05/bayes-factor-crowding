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

The current code includes an explicit mixed discrete-continuous model target for a sampler:

```text
For k = 1, ..., K:

  A_k ~ Dirichlet(alpha_k)

  Sigma_k ~ InverseWishart(nu_0, Psi_0)

  mu_k | Sigma_k ~ MultivariateNormal(m_0, Sigma_k / kappa_0)


Initial state:

  z_1 ~ Categorical(pi_0)


For t = 2, ..., T:

  z_t | z_{t-1}, A ~ Categorical(A[z_{t-1}])


For t = 1, ..., T:

  r_t | z_t = k, mu_k, Sigma_k
    ~ MultivariateStudentT(nu, mu_k, Sigma_k)


Constraint:

  trace(Sigma_1) <= trace(Sigma_2) <= ... <= trace(Sigma_K)
```

The implemented target is `models.student_t_regime.log_posterior`. It keeps the discrete path, transition matrix, regime mean vectors, and regime-specific full covariance matrices explicit so custom mixed discrete-continuous inference can sample them jointly.

## Inference Methodology

Inference is designed around a proprietary efficient mixed-tree variational parallel tempering algorithm. The method targets high-dimensional, multimodal, and strongly correlated mixed discrete-continuous posteriors, where the latent regime path is discrete and regime parameters are continuous.

The public codebase exposes the model target and integration boundary, while the core sampler implementation is kept separate. Conceptually, the inference engine samples from the joint posterior over regime paths, transition probabilities, regime mean vectors, and regime covariance matrices, then reports posterior regime probabilities for downstream risk overlays.

## Layout

```text
src/bayes_factor_crowding/
  data/        load and validate factor returns
  features/    build leakage-free model inputs
  models/      explicit latent-regime log posterior and configuration
  inference/   stable interfaces plus mixed-tempering adapter placeholder
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
