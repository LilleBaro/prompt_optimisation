from datasets import load_dataset

def load_gsm8k(
    train_size=200,
    validation_size=200,
    seed=42
):

    """
    Load and split GSM8K into optimization and validation subsets.

    Parameters
    ----------
    train_size : float, default=200
        Number of examples used by the RL agent during optimization.
    
    validation_size : float, default=200
        Number of examples used to evaluate candidate prompts
        during development.
    
    seed : float, default=42
        Random seed used for reproducible sampling.
    
    Returns
    -------
    dict 
        Dictionary containing:
        - `optimization` : subset used by the RL agent;
        - `validation`: subset used for model selection;
        - `test`: official GSM8K test set.
    """

    dataset = load_dataset(
        "openai/gsm8k",
        "main"
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.shuffle(
        seed=seed
    )

    # Split optimization/validation
    
    required_size = (
        train_size + validation_size
    )

    if required_size > len(train_dataset):

        raise ValueError(
            f"Requested {required_size} examples,",
            f"but GSM8K train contains only"
            f"{len(train_dataset)} examples."
        )
    optimization_dataset = train_dataset.select(
        range(
            train_size,
        )
    )
    validation_dataset = train_dataset.select(
        range(
            train_size,
            train_size + validation_size
        )
    )
    return {
        "optimization": optimization_dataset,
        "validation": validation_dataset,
        "test" : test_dataset
    }