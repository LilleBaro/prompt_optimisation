import re


class GSM8KEvaluator:
    """
    Evaluate a frozen LLM on GSM8K problems.

    The evaluator supports:

    - prompt-level caching;
    - batched LLM generation;
    - accuracy computation;
    - evaluation statistics.

    Parameters
    ----------
    llm : object
        Frozen language model exposing ``generate(prompt)``
        and ``generate_batch(prompts)`` methods.

    dataset : iterable
        GSM8K dataset containing ``question`` and ``answer`` fields.

    batch_size : int, default=8
        Number of GSM8K problems evaluated simultaneously.
    """

    def __init__(
        self,
        llm,
        dataset,
        batch_size=8,
    ):
        self.llm = llm
        self.dataset = dataset
        self.batch_size = batch_size

        # ---------------------------------------------------------
        # Prompt-level cache
        # ---------------------------------------------------------

        self.cache = {}

        # ---------------------------------------------------------
        # Evaluation statistics
        # ---------------------------------------------------------

        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_calls = 0

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

            Otherwise, a dictionary containing accuracy,
            individual results and evaluation statistics.
        """

        # ---------------------------------------------------------
        # Prompt-level cache
        # ---------------------------------------------------------

        if prompt in self.cache:

            self.cache_hits += 1

            cached_result = self.cache[prompt]

            if return_details:
                return cached_result

            return cached_result["accuracy"]

        # ---------------------------------------------------------
        # Cache miss
        # ---------------------------------------------------------

        self.cache_misses += 1

        # ---------------------------------------------------------
        # Build all GSM8K prompts
        # ---------------------------------------------------------

        examples = list(self.dataset)

        prompts = [
            self.build_prompt(
                prompt,
                example["question"],
            )
            for example in examples
        ]

        # ---------------------------------------------------------
        # Batched generation
        # ---------------------------------------------------------

        responses = []

        for start in range(
            0,
            len(prompts),
            self.batch_size,
        ):

            batch_prompts = prompts[
                start:start + self.batch_size
            ]

            batch_responses = (
                self.llm.generate_batch(
                    batch_prompts
                )
            )

            responses.extend(
                batch_responses
            )

            self.llm_calls += len(
                batch_prompts
            )

        # ---------------------------------------------------------
        # Evaluate predictions
        # ---------------------------------------------------------

        results = []

        for example, response in zip(
            examples,
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

        accuracy = (
            sum(
                result["correct"]
                for result in results
            )
            / len(results)
        )

        # ---------------------------------------------------------
        # Cache complete result
        # ---------------------------------------------------------

        evaluation_result = {
            "accuracy": accuracy,
            "results": results,
        }

        self.cache[prompt] = evaluation_result

        if return_details:
            return evaluation_result

        return accuracy

    # =============================================================
    # STATISTICS
    # =============================================================

    def get_statistics(self):
        """
        Return evaluator statistics.

        Returns
        -------
        dict
            Evaluation and cache statistics.
        """

        total_evaluations = (
            self.cache_hits
            + self.cache_misses
        )

        return {
            "total_evaluations": total_evaluations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "llm_calls": self.llm_calls,
            "cache_hit_rate": (
                self.cache_hits
                / total_evaluations
                if total_evaluations > 0
                else 0.0
            ),
        }

    # =============================================================
    # PROMPT BUILDING
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
        Extract the final numerical answer from a GSM8K target.

        Parameters
        ----------
        answer : str
            Original GSM8K answer containing reasoning and
            final answer.

        Returns
        -------
        str
            Normalized final answer.
        """

        match = re.search(
            r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
            answer,
        )

        if match is None:

            raise ValueError(
                f"Could not extract GSM8K target answer: "
                f"{answer}"
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

        Parameters
        ----------
        response : str
            Generated model response.

        Returns
        -------
        str or None
            Extracted numerical answer.
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
        # Fallback: last number
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
        Normalize a numerical answer.

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

    # =============================================================
    # CORRECTNESS
    # =============================================================

    @staticmethod
    def is_correct(
        prediction,
        target,
    ):
        """
        Compare a prediction with the GSM8K target.

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