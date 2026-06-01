---
layout: distill
title: "Optimizers Are Memory: A Visual Guide from SGD to AdamW"
description: "A visual guide to SGD, Momentum, RMSProp, Adam, and AdamW as different memory mechanisms over gradients."
date: 2026-06-01
tags: [deep-learning, optimization, pytorch, optimizers, adamw]
giscus_comments: true
---

## 1. Why optimizers matter

Most training scripts eventually land on a line like this:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

It is easy to treat this as plumbing. But that one line decides how every parameter moves.

Backpropagation gives gradients. The optimizer decides what to do with them.

A gradient answers a local question:

> If I nudge this parameter right now, which direction reduces the loss fastest?

An optimizer answers a richer question:

> Given this gradient and what I remember from earlier gradients, how should I update now?

That memory view is the thesis of this post:

- SGD remembers nothing.
- Momentum remembers direction.
- RMSProp remembers gradient scale.
- Adam remembers direction and scale.
- AdamW keeps Adam's memories but decouples weight decay from adaptive gradient scaling.

This is not a benchmark post. The learning rates here are tuned for visual clarity and stable convergence, not leaderboard comparisons.

## 2. SGD: no memory

SGD is the simplest update rule:

$$
w_{t+1} = w_t - \eta g_t
$$

No optimizer state. No moving averages. No memory.

```python
class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g
```

On a round quadratic bowl,

$$
f(x, y) = \frac{1}{2}(x^2 + y^2), \quad \nabla f(x, y) = [x, y]
$$

the current gradient is already a very good guide.

![SGD on a simple bowl](/assets/images/optimizers-memory/sgd_bowl.png)

*On a round bowl, the current gradient is a reliable guide. SGD does not need memory when the landscape is this friendly.*

## 3. When SGD zig-zags

Now change the geometry:

$$
f(x, y) = \frac{1}{2}(x^2 + 50y^2), \quad \nabla f(x, y) = [x, 50y]
$$

The surface is still convex, but curvature is very different by direction. In the steep coordinate, SGD keeps reacting to the latest large gradient, often crossing the valley and bouncing back.

So the issue is not that gradients are wrong. They are locally correct. The issue is that SGD has no memory of persistent vs. alternating directions.

## 4. Momentum: direction memory

Momentum adds a velocity state:

$$
v_t = \beta v_{t-1} + g_t, \quad w_{t+1} = w_t - \eta v_t
$$

```python
class Momentum:
    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = None
```

A useful mental model is a low-pass filter over gradients: persistent directions accumulate, frequently flipping directions are damped.

![Momentum in a ravine](/assets/images/optimizers-memory/momentum_ravine.png)

*SGD keeps reacting to the latest sideways gradient. Momentum remembers the directions that persist and damps directions that flip.*

## 5. RMSProp: scale memory

RMSProp tracks gradient magnitude per coordinate:

$$
s_t = \beta s_{t-1} + (1-\beta)g_t^2
$$

$$
w_{t+1} = w_t - \eta \frac{g_t}{\sqrt{s_t} + \epsilon}
$$

So coordinates with consistently large gradients get smaller effective steps.

![RMSProp scale adaptation](/assets/images/optimizers-memory/rmsprop_scale.png)

*RMSProp remembers how large each coordinate's gradients have been. Coordinates with consistently large gradients get smaller effective steps.*

## 6. Adam: direction + scale memory

Adam combines both memories:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

and updates with bias-corrected moments:

$$
w_{t+1} = w_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

On noisy gradients, this often gives smoother directional behavior with coordinate-wise scaling.

![Adam on noisy gradients](/assets/images/optimizers-memory/adam_noisy.png)

*Adam combines direction memory and scale memory: the first moment smooths noisy directions, while the second moment rescales coordinates with consistently large gradients.*

## 7. Why bias correction exists

Moving averages start at zero, so they are biased low early in training.

For a constant gradient sequence with EMA,

$$
m_t = \beta m_{t-1} + (1-\beta)g_t
$$

Adam corrects startup bias with:

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$

![Bias correction](/assets/images/optimizers-memory/bias_correction.png)

*Moving averages initialized at zero underestimate early values. Bias correction removes this startup artifact.*

## 8. AdamW: decoupling regularization

With L2-style regularization inside Adam, the penalty term is mixed into Adam's adaptive gradient path. AdamW separates these concerns:

- Adam + L2-style penalty in gradients:

```python
grad = grad + weight_decay * param
adam_update(param, grad)
```

- AdamW decoupled weight decay:

```python
param = param * (1 - lr * weight_decay)
adam_update(param, grad)
```

So weight decay remains direct shrinkage of parameters.

![AdamW weight decay comparison](/assets/images/optimizers-memory/adamw_weight_decay.png)

*AdamW decays weights directly. Adam with L2 first mixes the penalty into the gradient, then sends it through Adam's adaptive denominator.*

## 9. Sidebar: the memory cost of remembering

For a model with $N$ parameters, optimizer state overhead is:

| Optimizer | Extra state values |
|---|---:|
| SGD | 0 |
| Momentum | $N$ |
| RMSProp | $N$ |
| Adam | $2N$ |
| AdamW | $2N$ |

This is literal memory cost, not just conceptual complexity.

## 10. Linear regression sanity check

We also train a simple linear model:

$$
y = Xw + b + \epsilon
$$

This task is intentionally boring; it is a correctness check for from-scratch implementations.

![Linear regression loss curves](/assets/images/optimizers-memory/linear_regression_loss.png)

*Linear regression is intentionally boring. It checks that every optimizer implementation can solve a simple differentiable problem.*

State-size printout from the run:

```text
Model parameters: 11
SGD state values: 0
Momentum state values: 11
RMSProp state values: 11
Adam state values: 22
AdamW state values: 22
```

## 11. Tiny neural network on make_moons

We train a small MLP (2 -> 32 -> 32 -> 1) with BCEWithLogitsLoss on `make_moons`.

Parameter count:

```text
1185
```

Final decision boundaries:

![Two-moons decision boundaries](/assets/images/optimizers-memory/moons_decision_boundary.png)

*The two-moons task is small enough to visualize, but nonlinear enough that optimizer choice changes how quickly and cleanly the model finds a useful boundary.*

Final accuracies from the generated run:

```text
SGD: train=0.9625, test=0.9700
Momentum: train=0.9650, test=0.9800
AdamW: train=0.9650, test=0.9900
```

This is still a small didactic experiment, not a universal ranking.

## 12. Practical takeaways

- SGD is the cleanest baseline: zero optimizer memory, strong when geometry is friendly and schedules are tuned.
- Momentum helps when gradients oscillate in one direction but remain consistent in another.
- RMSProp helps when gradient scales differ across coordinates.
- Adam combines directional smoothing and scale adaptation, which is why it is often forgiving.
- AdamW keeps Adam's adaptive behavior while making weight decay behavior cleaner.

A common practical default remains:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
```

## 13. What comes next

Natural extensions from this memory-first view:

- learning-rate warmup and cosine schedules
- Nesterov momentum
- AdaGrad
- modern optimizers such as Adafactor, Lion, Sophia, and 8-bit Adam
- parameter groups and layer-wise optimization policies

## Reproducing the figures

All figures in this post are generated from scratch with NumPy and PyTorch scripts:

```bash
python scripts/optimizers_memory/generate_all.py
```

The settings are tuned for stable, visually interpretable behavior.
