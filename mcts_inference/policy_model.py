from typing import List, Tuple
from vllm import LLM, SamplingParams

from config import MCTSConfig


class PolicyModel:
    """vLLM-backed policy model that generates 256-token reasoning chunks."""

    def __init__(self, config: MCTSConfig):
        self.config = config
        self.llm = LLM(
            model=config.policy_model_name,
            trust_remote_code=True,
            tensor_parallel_size=config.tensor_parallel_size,
            max_logprobs=5,
        )
        self.tokenizer = self.llm.get_tokenizer()

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, question: str, partial_reasoning: str = "") -> str:
        """Format into Qwen chat template.

        The partial_reasoning is appended as the beginning of the assistant
        response so vLLM continues from there.
        """
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": question},
        ]
        # Apply chat template up to assistant turn
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Append any partial reasoning already generated
        if partial_reasoning:
            prompt += partial_reasoning
        return prompt

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_continuations(
        self,
        question: str,
        partial_reasoning: str = "",
        n: int = 2,
    ) -> List[Tuple[str, List[float], bool]]:
        """Generate n continuations of up to max_tokens_per_node tokens.

        Returns:
            List of (text, logprobs, hit_stop) tuples.
            - text: the generated chunk
            - logprobs: per-token log probabilities
            - hit_stop: True if generation stopped due to EOS or stop string
        """
        prompt = self.format_prompt(question, partial_reasoning)

        params_kwargs = dict(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens_per_node,
            n=n,
            logprobs=1,  # we only need the chosen token's logprob
        )
        if self.config.stop_strings:
            params_kwargs["stop"] = self.config.stop_strings
            params_kwargs["include_stop_str_in_output"] = True
        sampling_params = SamplingParams(**params_kwargs)

        outputs = self.llm.generate([prompt], sampling_params)

        results = []
        for output in outputs[0].outputs:
            text = output.text
            token_count = len(output.token_ids)

            # Extract per-token logprobs
            lps: List[float] = []
            if output.logprobs:
                for token_lp in output.logprobs:
                    if token_lp:
                        # token_lp is a dict {token_id: Logprob}
                        for _, lp_obj in token_lp.items():
                            val = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
                            lps.append(val)
                            break  # only the chosen token

            # Did generation stop early (EOS or stop string)?
            hit_stop = output.finish_reason == "stop"

            results.append((text, lps, hit_stop))

        return results
