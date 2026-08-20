class ManualBaseline:
    """
    Baseline based on manually designed prompt transformation sequences.

    Parameters
    ----------
    env : PromptOptimizationEnv
        Prompt optimization environment.

    action_sequences : list[list[int]]
        List of manually designed transformation sequences.
    """

    def __init__(self, env, action_sequences):
        self.env = env
        self.action_sequences = action_sequences

    def run(self):
        """
        Evaluate all manually designed prompt configurations.

        Returns
        -------
        dict
            Best manually designed prompt and results for all configurations.
        """

        results = []

        for actions in self.action_sequences:

            self.env.reset()

            for action in actions:

                _, _, terminated, truncated, info = (
                    self.env.step(action)
                )

                if terminated or truncated:
                    break

            results.append({
                "actions": actions.copy(),
                "prompt": info["prompt"],
                "accuracy": info["accuracy"],
            })

        best_result = max(
            results,
            key=lambda x: x["accuracy"]
        )

        return {
            "best_prompt": best_result["prompt"],
            "best_accuracy": best_result["accuracy"],
            "best_actions": best_result["actions"],
            "results": results,
        }