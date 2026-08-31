# Prompt Optimization with Reinforcement Learning for GSM8K

## Project overview

This project investigates whether a Reinforcement Learning agent can learn to improve a frozen language model’s performance by selecting prompt modifications automatically. The goal is not to teach the model how to answer questions directly, but to learn which instruction-level transformation is most likely to improve the model’s reasoning quality on a given task.

The system is designed around the following principle:

- the LLM is frozen and used only as a response generator;
- the RL agent does not learn the final answer itself;
- the RL agent learns to choose which prompt transformation to apply to improve performance on a downstream reasoning task;
- the benchmark used is GSM8K, a dataset of grade-school math word problems.

The agent observes the current prompt state and selects transformations such as:

- solving the problem step by step,
- checking the calculations,
- identifying the relevant information,
- explaining the reasoning,
- verifying the final answer,
- enforcing a clear response format.

The reward is primarily based on the accuracy of the model on GSM8K examples, with possible penalties linked to prompt complexity or length.

## Scientific objective

The central question of this project is:

"Given a task, which prompt modification is most likely to improve the performance of a frozen LLM?"

In other words, the agent learns an instruction-optimization policy driven by reward feedback rather than by explicit handcrafted prompt engineering.

## Project architecture

The repository is organized as follows:

- `agent/` - RL agents, including PPO
- `environment/` - Gymnasium environment for prompt optimization
- `evaluation/` - correctness evaluation on GSM8K
- `models/` - frozen LLM wrapper
- `prompts/` - base prompt and prompt transformation library
- `baselines/` - baseline strategies such as manual prompting and random search
- `train_ppo.py` - main PPO training entry point

## Training environment

The experiments were carried out on cloud GPU resources, specifically using Kaggle and Google Colab environments equipped with T4 GPUs. This hardware configuration enabled iterative training and evaluation of the PPO-based prompt optimization policy within the available computational budget.

## Experimental results

The project already contains saved experimental outputs in the repository root:

- [baseline_results.json](baseline_results.json)
- [ppo_results.json](ppo_results.json)
- [ppo_results_40ep.json](ppo_results_40ep.json)

### Baseline and reference strategies

The file [baseline_results.json](baseline_results.json) contains the results for baseline methods.

Observed performance:

- best manual prompt accuracy: 0.44 (44%)
- best random-search prompt accuracy: 0.32 (32%)

This indicates that a manually designed prompt already performs notably better than a random search baseline on this 50-example evaluation subset.

### PPO results

The file [ppo_results.json](ppo_results.json) reports the first PPO experiment.

Observed performance:

- PPO accuracy: 0.14 (14%)

The additional file [ppo_results_40ep.json](ppo_results_40ep.json) reports a second PPO run with a longer training horizon.

Observed performance:

- PPO (40 episodes) accuracy: 0.22 (22%)

This second run improves over the initial PPO result, but it remains below the best manual baseline.

## Comparison of all evaluated strategies

| Strategy | Accuracy |
| --- | ---: |
| Manual prompt search | 0.44 |
| Random search | 0.32 |
| PPO (40 episodes) | 0.22 |
| PPO (initial run) | 0.14 |

This comparison shows that simple prompt engineering remains stronger than the current RL policy. The PPO agent improves with additional training episodes, but the gain is still insufficient to match the best handcrafted prompt.

## Analysis of the findings

The results suggest several important observations:

- the frozen LLM is already reasonably strong on GSM8K tasks;
- the RL agent has not yet identified reliable prompt modifications that consistently improve performance;
- the optimization signal remains noisy and difficult to exploit;
- exploration and reward design are critical factors in this setting.

This suggests that the challenge is not limited to prompt engineering alone. It also depends on the quality of the reward function, the stability of the environment, and the exploration strategy used during RL training.

## Key files

- [train_ppo.py](train_ppo.py) - PPO training script
- [agent/ppo.py](agent/ppo.py) - PPO implementation
- [environment/prompt_env.py](environment/prompt_env.py) - environment modeling prompt modifications
- [evaluation/gsm8k_evaluator.py](evaluation/gsm8k_evaluator.py) - GSM8K evaluation logic
- [models/frozen_llm.py](models/frozen_llm.py) - frozen LLM wrapper
- [prompts/base_prompt.py](prompts/base_prompt.py) - base instruction prompt
- [prompts/transformations.py](prompts/transformations.py) - prompt transformation definitions

## Local execution

To run the training script locally:

```bash
python train_ppo.py
```

## Local output convention

This project follows a local save convention for results, using workspace-relative paths instead of Kaggle-specific locations:

- `baseline_results.json`
- `ppo_results.json`
- `results/baseline_results.json`
- `results/ppo_results.json`

Hardcoded paths such as `/kaggle/working/...` should not be used in local runs.

## Discussion

The experimental results suggest that the RL-based prompt optimization framework is conceptually valid, but still limited in practice. The best handcrafted prompt attains 44% accuracy, while the strongest PPO configuration reaches only 22%. This gap indicates that the policy learned by the agent is not yet able to consistently discover prompt edits that outperform carefully designed human instructions.

Several factors may explain this behavior. First, the action space may be too small or too rigid to support high-quality prompt optimization. Second, the reward signal may be too sparse or unstable to guide learning effectively. Third, the environment may not provide enough informative state features to distinguish between prompt variants with comparable quality. Finally, the current training setup likely needs more episodes and stronger optimization stability to produce robust policy updates.

## Future work

Several directions are worth exploring in order to improve the approach:

- improve the reward design by combining task accuracy with additional shaping signals that better reflect incremental prompt usefulness;
- increase training stability by tuning PPO hyperparameters, clipping behavior, and learning-rate schedules;
- expand the set of available prompt transformations and allow more diverse combinations of instructions;
- evaluate the policy on larger GSM8K subsets and across multiple random seeds to reduce variance in the reported results;
- compare alternative RL formulations or stronger optimization methods to determine whether the bottleneck is in the policy learning itself or in the environment design.

## Conclusion

This project demonstrates a promising but still incomplete approach to automatic prompt optimization through Reinforcement Learning. The RL agent is able to act in prompt space and improve slightly with additional training, but it remains below the performance of manual prompt engineering in the current implementation.

The observed results indicate that the core idea is sound but the optimization pipeline still requires better reward modeling, stronger exploration, and more stable training dynamics. With these improvements, RL-based prompt optimization could become a competitive alternative to handcrafted prompting on reasoning benchmarks such as GSM8K.
