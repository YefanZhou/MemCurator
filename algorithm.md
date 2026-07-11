# RL Training Algorithm: Skill Curator with Frozen Executor

## Overview

We train a **skill curator** policy using reinforcement learning while keeping the **executor** (the problem-solving LLM) frozen. The curator learns to maintain a dynamic skill memory that helps the executor solve future problems more accurately.

## Setup

**Two agents, one trainable:**

- **Executor** $\pi_{\text{exec}}$ (frozen): A pre-trained LLM (Qwen3-8B) that answers math questions by retrieving and conditioning on skills in memory via BM25 retrieval.
- **Curator** $\pi_{\theta}$ (trainable): A policy that observes each problem-solving attempt and updates the skill memory through structured function calls (insert, update, delete).

**Skill memory** $\mathcal{M}$: A dynamic key-value store of reusable mathematical skills. Initialized empty at the start of each episode.

## Training Data

Each training instance is a **grouped problem sequence** $(q_1, a_1), \ldots, (q_T, a_T)$ where all questions share a common topic or theme (e.g., Ring Theory, combinatorics). Grouping ensures that skills learned from early questions are relevant for later ones.

## Episode Rollout

For each training batch, we sample $N$ problem sequences and run $K$ independent rollouts per sequence (GRPO). Each rollout proceeds as follows:

For $t = 1, \ldots, T$:

1. **Retrieval**: The executor retrieves the top-$k$ skills from $\mathcal{M}_t$ most relevant to $q_t$ using BM25.

2. **Execution**: The frozen executor $\pi_{\text{exec}}$ generates a prediction $\hat{a}_t$ conditioned on $q_t$ and the retrieved skills.

3. **Curation**: The curator $\pi_\theta$ receives a context consisting of:
   - The question $q_t$
   - The retrieved skills used
   - The executor's prediction $\hat{a}_t$
   - The correct answer $a_t$

   The curator generates a sequence of function calls to update $\mathcal{M}_t$, producing a new memory state $\mathcal{M}_{t+1}$.

## Reward Design

The reward for the curator's action at step $t$ is a composite of four signals:

$$r_t = \underbrace{r_t^{\text{acc}}}_{\text{accuracy}} + \lambda_c \underbrace{r_t^{\text{comp}}}_{\text{compression}} + \lambda_f \underbrace{r_t^{\text{call}}}_{\text{function call}} + \lambda_u \underbrace{r_t^{\text{use}}}_{\text{memory use}}$$

**Accuracy reward** $r_t^{\text{acc}}$ (shifted credit assignment):

$$r_t^{\text{acc}} = \mathbb{1}[\hat{a}_{t+1} = a_{t+1}], \quad r_T^{\text{acc}} = 0$$

The curator at step $t$ is rewarded based on how well the executor performs on the *next* question $q_{t+1}$, not the current one. This is because the curation at step $t$ updates the memory used at step $t+1$; the current question's outcome $\mathbb{1}[\hat{a}_t = a_t]$ is independent of $\pi_\theta$'s action at step $t$ (the memory was already fixed before curation). The last step receives zero reward since there is no subsequent question.

**Compression reward** $r_t^{\text{comp}}$:

$$r_t^{\text{comp}} = 1 - \frac{|\mathcal{M}_{t+1}|}{|c_t|}$$

where $|\mathcal{M}_{t+1}|$ is the token length of the updated memory and $|c_t|$ is the token length of the input context (question + prediction). This encourages the curator to distill concise, reusable skills rather than copying raw content verbatim.

**Function call reward** $r_t^{\text{call}}$: Binary reward for each syntactically valid and successfully executed tool call (insert/update/delete). Encourages the curator to produce well-formed memory operations.

**Memory use reward** $r_t^{\text{use}}$: A Qwen3-32B judge evaluates whether the executor's reasoning in step $t+1$ explicitly utilizes content from the updated memory $\mathcal{M}_{t+1}$. This closes the loop by rewarding curator actions that produce skills the executor actually applies.

## Policy Optimization

We optimize $\pi_\theta$ using **GRPO** (Group Relative Policy Optimization). For each position $t$ in the sequence, advantages are computed across the $K$ rollouts:

$$A_t^{(k)} = r_t^{(k)} - \frac{1}{K} \sum_{j=1}^{K} r_t^{(j)}$$

All $T$ curation steps across the sequence contribute to the policy gradient. The curator's parameters are updated to increase the probability of actions that yield higher relative advantage, with a KL divergence penalty to the reference policy to prevent over-optimization.

## Key Design Principle

The shifted credit assignment is essential for learning. Without it, the reward at step $t$ would reflect the quality of memory *before* the curator acted—identical across all $K$ rollouts (since all start with the same memory state)—yielding zero advantage and no gradient signal for early curation steps. By assigning $r_t^{\text{acc}} = \mathbb{1}[\hat{a}_{t+1} = a_{t+1}]$, each curation step receives a signal that directly reflects the impact of its own memory update on the executor's subsequent performance.
