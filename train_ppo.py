from datasets import load_dataset
from models.frozen_llm import FrozenLLM
from evaluation.gsm8k_evaluator import GSM8KEvaluator

from prompts.transformations import PROMPT_TRANSFORMATIONS, apply_transformation
from prompts.base_prompt import BASE_PROMPT

from environment.prompt_env import PromptOptimizationEnv

from agent.ppo import PPO

dataset = load_dataset(
    "openai/gsm8k",
    "main"
)

train_dataset = dataset["train"].select(
    range(50)
)

llm = FrozenLLM(max_new_tokens=256, 
                temperature=0.0)

evaluator = GSM8KEvaluator(
    llm,
    train_dataset
)

env = PromptOptimizationEnv(
    base_prompt=BASE_PROMPT,
    evaluator=evaluator,
    max_steps=5
)

agent = PPO(
    env=env,
    learning_rate=1e-4,
    update_epochs=4
)

history = agent.train(
    n_episodes=20
)
