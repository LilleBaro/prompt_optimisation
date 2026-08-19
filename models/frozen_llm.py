from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class FronzenLLM:
    """
    Wrapper around a frozen causal language model for prompt optimization.

    The model is loaded in inference mode and its parameters are never updated.
    The class is responsible only ofr loading the model and generating responses.

    Parameters
    ------
    model_name : str, default="Qwen/Qwen2.5-3B-Instruct"
        Hugging Face model identifier.

    max_new_tokens : int, default=512
        Maximum number of tokens generated for each responses.

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
            device_map="auto"
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        # frozen model
        self.model.eval()
        for parameters in self.model.parameters():
            parameters.requires_grad = False

    def generate(self, prompt):
        """
        Generate a response for a given prompt.

        Parameters
        ------
        prompt : str
            Input prompt

        Returns
        ------
        str
            Generated response.
        """
        messages = [
            {
                "role":"user",
                "content":prompt
            }
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
        }

        if self.temperature > 0:
            generation_kwargs.update({
                "do_sample":True,
                "temperature": self.temperature,
            })
        else:
            generation_kwargs.update({
                "do_sample":False
            })
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        return response.strip()