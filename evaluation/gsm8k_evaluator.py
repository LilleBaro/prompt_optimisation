import re


class GSM8KEvaluator:
    """
    Evaluate a frozen LLM on GSM8K problems.

    Parameters
    ----------
    llm : object
        Frozen language model exposing a ``generate(prompt)`` method.

    dataset : iterable
        GSM8K dataset containing ``question`` and ``answer`` fields.
    """

    def __init__(self, llm, dataset):
        self.llm = llm
        self.dataset = dataset
        self.cache = {}

    def evaluate(self, prompt, return_details=False):
        """
        Evaluate a prompt on the GSM8K dataset.

        Parameters
        ----------
        prompt : str
            Prompt instruction to evaluate.

        return_details : bool, default=False
            If True, return detailed results for each problem.

        Returns
        -------
        float or dict
            Accuracy if ``return_details=False``.
            Otherwise, a dictionary containing accuracy and individual results.
        """

        if prompt in self.cache and not return_details:
            return self.cache[prompt]
        results = []

        for example in self.dataset:

            question = example["question"]
            target = self.extract_target_answer(
                example["answer"]
            )

            full_prompt = self.build_prompt(
                prompt,
                question
            )

            response = self.llm.generate(
                full_prompt
            )

            prediction = self.extract_prediction(
                response
            )

            correct = self.is_correct(
                prediction,
                target
            )

            results.append({
                "question": question,
                "target": target,
                "response": response,
                "prediction": prediction,
                "correct": correct
            })

        accuracy = (
            sum(result["correct"] for result in results)
            / len(results)
        )
        self.cache[prompt]=accuracy

        if return_details:
            return {
                "accuracy": accuracy,
                "results": results
            }

        return accuracy

    @staticmethod
    def build_prompt(prompt, question):
        """
        Combine the current instruction with a GSM8K question.

        Parameters
        ----------
        prompt : str
            Current optimized instruction.

        question : str
            GSM8K problem.

        Returns
        -------
        str
            Complete prompt sent to the LLM.
        """

        return f"""{prompt}

Problem:
{question}
"""

    @staticmethod
    def extract_target_answer(answer):
        """
        Extract the final numerical answer from a GSM8K target answer.

        Parameters
        ----------
        answer : str
            Original GSM8K answer containing the reasoning and final answer.

        Returns
        -------
        str
            Extracted final answer.
        """

        match = re.search(
            r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
            answer
        )

        if match is None:
            raise ValueError(
                f"Could not extract GSM8K target answer: {answer}"
            )

        return GSM8KEvaluator.normalize_number(
            match.group(1)
        )

    @staticmethod
    def extract_prediction(response):
        """
        Extract a numerical prediction from an LLM response.

        Parameters
        ----------
        response : str
            Generated response from the LLM.

        Returns
        -------
        str or None
            Extracted final numerical answer.
        """

        # explicit final-answer patterns.
        patterns = [
            r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
            r"(?:final answer|answer)\s*(?:is|:)?\s*([-+]?\d[\d,]*(?:\.\d+)?)"
        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                response,
                flags=re.IGNORECASE
            )

            if matches:
                return GSM8KEvaluator.normalize_number(
                    matches[-1]
                )

        # fallback:
        # use the last number appearing in the response.
        numbers = re.findall(
            r"[-+]?\d[\d,]*(?:\.\d+)?",
            response
        )

        if not numbers:
            return None

        return GSM8KEvaluator.normalize_number(
            numbers[-1]
        )

    @staticmethod
    def normalize_number(value):
        """
        Normalize a numerical answer before comparison.

        Parameters
        ----------
        value : str
            Numerical value.

        Returns
        -------
        str
            Normalized numerical representation.
        """

        value = value.strip()
        value = value.replace(",", "")

        try:
            number = float(value)

            if number.is_integer():
                return str(int(number))

            return str(number)

        except ValueError:
            return value

    @staticmethod
    def is_correct(prediction, target):
        """
        Compare a model prediction with a GSM8K target.

        Parameters
        ----------
        prediction : str or None
            Extracted model prediction.

        target : str
            Expected answer.

        Returns
        -------
        bool
            Whether the prediction is correct.
        """

        if prediction is None:
            return False

        return prediction == target