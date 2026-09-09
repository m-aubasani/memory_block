import os
import sys
import torch
from typing import List, Union, Optional, Dict, Any

# Ensure stdout uses UTF-8 to prevent encoding errors on non-UTF8 console environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class GLiGuardChecker:
    """
    Wrapper for GLiGuard (fastino/gliguard-LLMGuardrails-300M) safety classifier.
    Supports single and batched evaluation for prompt and response safety.
    """

    def __init__(
        self,
        model_name: str = "fastino/gliguard-LLMGuardrails-300M",
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.5,
    ):
        from gliner2 import GLiNER2

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        print(f"Loading GLiGuard Classifier ({model_name}) on {self.device}...")
        self.model = GLiNER2.from_pretrained(model_name)
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        self.threshold = threshold

        # Default schemas
        self.prompt_schema = {
            "prompt_safety": ["safe", "unsafe"],
        }
        self.response_schema = {
            "response_safety": ["safe", "unsafe"],
            "response_refusal": ["refusal", "compliance"],
        }

    def classify_prompts(
        self,
        prompts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Classifies prompt safety for a single prompt or list of prompts.
        """
        is_single = isinstance(prompts, str)
        prompt_list = [prompts] if is_single else list(prompts)

        if not prompt_list:
            return []

        if len(prompt_list) == 1:
            res = self.model.classify_text(
                prompt_list[0],
                self.prompt_schema,
                threshold=self.threshold,
            )
            return [res]

        results = self.model.batch_classify_text(
            prompt_list,
            self.prompt_schema,
            batch_size=batch_size,
            threshold=self.threshold,
        )
        return results

    def is_prompt_safe(
        self,
        prompts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> Union[bool, List[bool]]:
        """
        Returns True if prompt is safe, False otherwise.
        """
        is_single = isinstance(prompts, str)
        results = self.classify_prompts(prompts, batch_size=batch_size)
        verdicts = [res.get("prompt_safety") == "safe" for res in results]
        return verdicts[0] if is_single else verdicts

    def is_prompt_unsafe(
        self,
        prompts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> Union[bool, List[bool]]:
        """
        Returns True if prompt is unsafe, False otherwise.
        """
        is_single = isinstance(prompts, str)
        results = self.classify_prompts(prompts, batch_size=batch_size)
        verdicts = [res.get("prompt_safety") == "unsafe" for res in results]
        return verdicts[0] if is_single else verdicts

    def classify_responses(
        self,
        responses: Union[str, List[str]],
        prompts: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Classifies response safety for responses.
        Formats input with 'Prompt: ...\\nResponse: ...' if prompt is provided,
        or 'Response: ...' otherwise.
        """
        is_single = isinstance(responses, str)
        response_list = [responses] if is_single else list(responses)

        if not response_list:
            return []

        if prompts is not None:
            prompt_list = [prompts] * len(response_list) if isinstance(prompts, str) else list(prompts)
            formatted_texts = [
                f"Prompt: {p}\nResponse: {r}" for p, r in zip(prompt_list, response_list)
            ]
        else:
            formatted_texts = [f"Response: {r}" for r in response_list]

        if len(formatted_texts) == 1:
            res = self.model.classify_text(
                formatted_texts[0],
                self.response_schema,
                threshold=self.threshold,
            )
            return [res]

        results = self.model.batch_classify_text(
            formatted_texts,
            self.response_schema,
            batch_size=batch_size,
            threshold=self.threshold,
        )
        return results

    def is_response_safe(
        self,
        responses: Union[str, List[str]],
        prompts: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
    ) -> Union[bool, List[bool]]:
        """
        Determines whether response is safe according to GLiGuard decision rules:
        A response is considered safe if:
          - response_safety is 'safe', OR
          - response_refusal is 'refusal' (a refusal successfully prevents harm).
        """
        is_single = isinstance(responses, str)
        results = self.classify_responses(responses, prompts=prompts, batch_size=batch_size)
        verdicts = []
        for res in results:
            resp_safe = res.get("response_safety") == "safe"
            is_refusal = res.get("response_refusal") == "refusal"
            verdicts.append(resp_safe or is_refusal)

        return verdicts[0] if is_single else verdicts
