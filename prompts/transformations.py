PROMPT_TRANSFORMATIONS = {

    0: {
        "name": "step_by_step",
        "instruction":"Think step by step."},
    1: {
        "name":"reasoning",
        "instruction":"Solve the problem step by step"},
    2: {
        "name":"verification",
        "instruction":"Verify your answer before giving the final answer."},
    3: {
        "name":"calculation_check",
        "instruction":"Check your calculations carefully."},
    4: {
        "name":"decomposition",
        "instruction":"Explain your reasoning. clearly"},
    5: {
        "name":"relevant_information",
        "instruction":"Identify the relevant information before solving the problem."},
    6: {
        "name":"double_check",
        "instruction":"Breack the problem into smaller steps."},
    7: {
        "name":"stop",
        "instruction":"STOP"
    }

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
        Transformed prompt
    """
    transformation = PROMPT_TRANSFORMATIONS[action]["instruction"]

    if not transformation:
        return prompt
    return f"{prompt}\n\n{transformation}"