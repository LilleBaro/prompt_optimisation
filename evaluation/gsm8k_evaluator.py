import re


class GSM8KEvaluator:
    """
    Evaluate a frozen LLM on GSM8K problems.

    Parameters
    ----------
    llm : object
        Frozen language model exposing ``generate`` and
        ``generate_batch`` methods.

    dataset : iterable
        GSM8K dataset containing ``question`` and ``answer`` fields.

    batch_size : int, default=8
        Number of GSM8K problems processed simultaneously.
    """

    def __init__(
        self,
        llm,
        dataset,
        batch_size=8,
    ):
        self.llm = llm
        self.dataset = list(dataset)
        self.batch_size = batch_size

        # Cache:
        # prompt -> accuracy
        self.cache = {}

    # =============================================================
    # EVALUATION
    # =============================================================

    def evaluate(
        self,
        prompt,
        return_details=False,
    ):
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

            Otherwise, a dictionary containing accuracy
            and individual results.
        """

        # ---------------------------------------------------------
        # Cache
        # ---------------------------------------------------------

        if (
            prompt in self.cache
            and not return_details
        ):
            return self.cache[prompt]

        # ---------------------------------------------------------
        # Build all prompts
        # ---------------------------------------------------------

        full_prompts = [
            self.build_prompt(
                prompt,
                example["question"],
            )
            for example in self.dataset
        ]

        results = []

        # ---------------------------------------------------------
        # Batch inference
        # ---------------------------------------------------------

        for start in range(
            0,
            len(full_prompts),
            self.batch_size,
        ):

            end = start + self.batch_size

            prompt_batch = full_prompts[
                start:end
            ]

            examples_batch = self.dataset[
                start:end
            ]

            responses = self.llm.generate_batch(
                prompt_batch
            )

            # -----------------------------------------------------
            # Evaluate batch
            # -----------------------------------------------------

            for example, response in zip(
                examples_batch,
                responses,
            ):

                question = example["question"]

                target = self.extract_target_answer(
                    example["answer"]
                )

                prediction = self.extract_prediction(
                    response
                )

                correct = self.is_correct(
                    prediction,
                    target,
                )

                results.append({
                    "question": question,
                    "target": target,
                    "response": response,
                    "prediction": prediction,
                    "correct": correct,
                })

        # ---------------------------------------------------------
        # Accuracy
        # ---------------------------------------------------------

        if not results:
            raise ValueError(
                "The GSM8K dataset is empty."
            )

        accuracy = (
            sum(
                result["correct"]
                for result in results
            )
            / len(results)
        )

        # ---------------------------------------------------------
        # Cache accuracy
        # ---------------------------------------------------------

        self.cache[prompt] = accuracy

        # ---------------------------------------------------------
        # Return
        # ---------------------------------------------------------

        if return_details:
            return {
                "accuracy": accuracy,
                "results": results,
            }

        return accuracy

    # =============================================================
    # PROMPT CONSTRUCTION
    # =============================================================

    @staticmethod
    def build_prompt(
        prompt,
        question,
    ):
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

    # =============================================================
    # TARGET EXTRACTION
    # =============================================================

    @staticmethod
    def extract_target_answer(answer):
        """
        Extract the final numerical answer from a GSM8K target answer.
        """

        match = re.search(
            r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
            answer,
        )

        if match is None:
            raise ValueError(
                f"Could not extract GSM8K target answer: {answer}"
            )

        return GSM8KEvaluator.normalize_number(
            match.group(1)
        )

    # =============================================================
    # PREDICTION EXTRACTION
    # =============================================================

    @staticmethod
    def extract_prediction(response):
        """
        Extract a numerical prediction from an LLM response.
        """

        patterns = [
            r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",

            r"(?:final answer|answer)"
            r"\s*(?:is|:)?\s*"
            r"([-+]?\d[\d,]*(?:\.\d+)?)",
        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                response,
                flags=re.IGNORECASE,
            )

            if matches:

                return GSM8KEvaluator.normalize_number(
                    matches[-1]
                )

        # ---------------------------------------------------------
        # Fallback
        # ---------------------------------------------------------

        numbers = re.findall(
            r"[-+]?\d[\d,]*(?:\.\d+)?",
            response,
        )

        if not numbers:
            return None

        return GSM8KEvaluator.normalize_number(
            numbers[-1]
        )

    # =============================================================
    # NUMBER NORMALIZATION
    # =============================================================

    @staticmethod
    def normalize_number(value):
        """
        Normalize a numerical answer before comparison.
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

    # =============================================================
    # CORRECTNESS
    # =============================================================

    @staticmethod
    def is_correct(
        prediction,
        target,
    ):
        """
        Compare a model prediction with a GSM8K target.
        """

        if prediction is None:
            return False

        return prediction == target