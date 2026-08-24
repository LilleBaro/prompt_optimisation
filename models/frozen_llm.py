from transformers import AutoTokenizer, AutoModelForCausalLM

import torch


class FrozenLLM:
    """
    Wrapper around a frozen causal language model for prompt optimization.

    The model is loaded in inference mode and its parameters are never
    updated. The class is responsible only for loading the model and
    generating responses.

    Parameters
    ----------
    model_name : str, default="Qwen/Qwen2.5-3B-Instruct"
        Hugging Face model identifier.

    max_new_tokens : int, default=512
        Maximum number of tokens generated for each response.

    temperature : float, default=0.0
        Sampling temperature. A value of 0 uses deterministic generation.

    device_map : str, default="auto"
        Device mapping used to load the model.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=512,
        temperature=0.0,
        device_map="auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # ---------------------------------------------------------
        # Tokenizer
        # ---------------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        # Qwen may not have a pad token explicitly defined.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        # ---------------------------------------------------------
        # Freeze model
        # ---------------------------------------------------------

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    # =============================================================
    # GENERATION CONFIGURATION
    # =============================================================

    def _generation_kwargs(self):
        """
        Build generation parameters.

        Returns
        -------
        dict
            Parameters passed to ``model.generate``.
        """

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": self.temperature,
            })
        else:
            generation_kwargs.update({
                "do_sample": False,
            })

        return generation_kwargs

    # =============================================================
    # SINGLE GENERATION
    # =============================================================

    def generate(self, prompt):
        """
        Generate a response for a single prompt.

        Parameters
        ----------
        prompt : str
            Input prompt.

        Returns
        -------
        str
            Generated response.
        """

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                **self._generation_kwargs(),
            )

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[0][
            input_length:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response.strip()

    # =============================================================
    # BATCH GENERATION
    # =============================================================

    def generate_batch(self, prompts):
        """
        Generate responses for multiple prompts simultaneously.

        Parameters
        ----------
        prompts : list[str]
            List of input prompts.

        Returns
        -------
        list[str]
            Generated responses in the same order as the input prompts.
        """

        if not prompts:
            return []

        messages = [
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            for prompt in prompts
        ]

        # ---------------------------------------------------------
        # Apply chat template to the entire batch
        # ---------------------------------------------------------

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        # ---------------------------------------------------------
        # Generate
        # ---------------------------------------------------------

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                **self._generation_kwargs(),
            )

        # ---------------------------------------------------------
        # Extract only generated tokens
        # ---------------------------------------------------------

        input_lengths = (
            inputs["attention_mask"]
            .sum(dim=1)
        )

        responses = []

        for output, input_length in zip(
            outputs,
            input_lengths,
        ):

            generated_tokens = output[
                input_length:
            ]

            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            responses.append(
                response.strip()
            )

        return responses