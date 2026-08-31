import random

class RandomSearch:
    """
    Random search baseline for prompt optimization.
    
    The algorithm randomly selects prompt transformations and evaluates
    the resulting prompt. The best-performing prompt found during the 
    search is returned

    Parameters
    ----------
    env : PromptOptimizationEnv
        Prompt optimization environment

    n_trials : int, default=20
        Number of random prompt configuration to evaluate.
    
    max_steps : int, default=3
        Maximum number of transformations in a single trial.
    
    seed : int or None, default=None
        Random seed for reproducibility.
    """ 

    def __init__(
            self,
            env,
            n_trials=20,
            max_steps=3,
            seed=None
    ):
        self.env=env
        self.n_trials=n_trials
        self.max_steps=max_steps
        if seed is not None:
            random.seed(seed)

    def run(self):
        """
        Execute the random search
        Returns
        -------
        dict
            Best prompt and associated information.
        """

        best_prompt = None
        best_accuracy = float("-inf")
        best_action= None

        trials_results = []
        for trial in range(self.n_trials):

            observation, info = self.env.reset()
            available_actions = list(
                range(self.env.action_dim)
                )
            selected_actions = []

            # random number of transformations
            n_steps = random.randint(
                1,
                self.max_steps,
            )

            for _ in range(n_steps):
                # select only transformation 
                # that have not already been used
                remaining_actions = [
                    action for action in available_actions
                    if action not in selected_actions
                ]

                if not remaining_actions:
                    break

                action = random.choice(
                    remaining_actions
                )
                selected_actions.append(action)
                observation, reward, terminated, truncated, info = (
                    self.env.step(action)
                )

                if terminated or truncated:
                    break

            accuracy = info['accuracy']
            result= {
                "trial": trial,
                "prompt": info["prompt"],
                "accuracy": accuracy,
                "actions": selected_actions.copy(),
            }

            trials_results.append(result)

            if accuracy > best_accuracy:
                best_accuracy=accuracy
                best_prompt=info["prompt"]
                best_action=selected_actions.copy()
        return {
                "best_prompt": best_prompt,
                "best_accuracy": best_accuracy,
                "best_action": best_action,
                "trials": trials_results
            }
    

