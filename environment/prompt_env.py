
PROMPT_TRANSFORMATION = {
    0: "",
    1: "Think step by step.",
    2: "Solve the problem step by step",
    3: "Verify your answer before giving the final answer.",
    4: "Check your calculations carefully.",
    5: "Explain your reasoning. clearly",
    6: "Identify the relevant information before solving the problem.",
    7: "Breack the problem into smaller steps.",
    8: "Double-check your final answer.",
    9: "Provide hte final answer clearly.",
}

def apply_transformation(prompt, action):
    """
    Apply a prompt transformation.
    args
    -------
    prompt : str
        current prompt.
    action : int
        ID of the transformation to apply
    Returns
    -------
    str
        Updated prompt
    """
    transformation = PROMPT_TRANSFORMATION[action]

    if not transformation:
        return prompt
    return f"{prompt}\n\n{transformation}"