def compute_accuracy_gain(
        accuracy,
        baseline_accuracy
):
    """
    Compute the absolute accuracy improvement over a baseline.
    
    Parameters
    ----------
    accuracy : float
        Accuracy obtained by the evaluated prompt.
    
    baseline_accuracy
        Accuracy obtained by the baseline prompt.
    
    Returns
    -------
    float
        Absolute accuracy gain.
    """
    return accuracy - baseline_accuracy

def compute_relative_gain(
        accuracy,
        baseline_accuracy
):
    """
    Compute the relative accuracy improvement over a baseline.
    
    Parameters
    ----------
    accuracy : float
        Accuracy obtained by the evaluated prompt.
    
    baseline accuracy 
        Accuracy obtained by the baseline prompt.

    Returns
    -------
    float 
        Relative accuracy gain.
    """
    if baseline_accuracy==0:
        return 0.0
    return (
        (accuracy - baseline_accuracy)/baseline_accuracy
    )

def compute_prompt_length(prompt):
    """
    Compute the nulber of words in a prompt.
    
    Parameters
    ----------
    prompt : str
        Prompt to evaluate.
    
    Returns
    -------
    int 
        Number of words in the prompt.
    """
    return len(prompt.split())

def compute_reward(
        current_accuracy,
        previous_accuracy):
    """
    Compute the reward associated with an accuracy improvement.

    Parameters
    ----------
    current_accuracy : float
        Accuracy adter the action.
    
    previous accuracy : float
        Accuracy before the actions.
    
    Returns
    -------
    float
        Accuracy improvement.
    """
    return current_accuracy - previous_accuracy

def build_metrics(
        accuracy,
        baseline_accuracy,
        prompt,
        actions=None,
        cumulative_reward=None
):
    """
    Build a standardized metrics dictionnary.
    
    Parameters
    ----------
    accuracy : float
        Accuracy of the evaluated prompt.
    
    baseline_accracy : float:
        Accuracy of the base prompt
        
    prompt : str
        Evaluated prompt.
        
    actions : list[int]
        Transformations used to construct the prompt.
        
    cumulative_reward : flaot or None
        Total reward accumulated during the episode.

    Returns 
    -------
    dict 
        Standardized evaluation metrics.
    """

    if actions is None:
        actions = []
    return {
        "accuracy": accuracy,
        "accuracy_gain":compute_accuracy_gain(
            accuracy, baseline_accuracy
        ),
        "relative_gain": compute_relative_gain(
            accuracy, baseline_accuracy
        ),
        "prompt_length": compute_prompt_length(
            prompt
        ),
        "n_transformations": len(actions),
        "cumulative_reward": cumulative_reward
    }
