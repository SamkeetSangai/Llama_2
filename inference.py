# Standard Library Imports
import time
import json
from pathlib import Path

# Typing Imports
from typing import Optional

# Third-Party Library Imports
import torch
from tqdm import tqdm
from sentencepiece import SentencePieceProcessor

# Local Module Imports
from model import ModelArgs, Transformer


class LLaMA:
    """
    LLaMA (Large Language Model Meta AI) class for loading and generating text completions.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs,
    ):
        """
        Initializes the LLaMA model with a transformer model, tokenizer, and model arguments.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        """
        Builds the LLaMA model by loading the transformer, tokenizer, and model parameters.
        """
        prev_time = time.time()

        # Load checkpoint if required
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert (
                len(checkpoints) > 0
            ), f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()

        # Load model parameters from JSON file
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )

        # Load tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # Set default tensor type based on device
        torch.set_default_dtype(torch.float16 if device == "cuda" else torch.bfloat16)
        torch.set_default_device(device)

        # Initialize model
        model = Transformer(model_args).to(device)

        if load_model:
            # Remove the only unmatched key "rope.freqs" before loading the state dict
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return LLaMA(model, tokenizer, model_args)

    def text_completion(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):
        """
        Generates text completions given a list of input prompts.
        """
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # Convert each prompt into tokens
        prompt_tokens = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]

        batch_size = len(prompt_tokens)
        assert (
            batch_size <= self.args.max_batch_size
        ), f"batch size must be <= {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert (
            max_prompt_len <= self.args.max_seq_len
        ), f"prompt length must be <= {self.args.max_seq_len}"

        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create tensor for storing tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=device
        )
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id  # Mask for prompt tokens

        # Iterate over token positions and generate new tokens
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            if all(eos_reached):
                break

        # Decode output tokens
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        """
        Performs Top-p (nucleus) sampling.
        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = [
        # Zero-shot
        "Explain the concept of quantum entanglement in simple terms.",
        # One-shot
        """If Apple was founded in Japan instead of the US, it would have likely focused on:
        Example: Sony revolutionized consumer electronics with the Walkman.
        Apple, in a Japanese context, would have likely...""",
        # Few-shot
        """Translate English to French:
        sunflower => tournesol
        laptop => ordinateur portable
        wireless headphones => écouteurs sans fil
        jellyfish =>""",
    ]

    model = LLaMA.build(
        checkpoints_dir="Llama-2-7b-chat",
        tokenizer_path="Llama-2-7b-chat/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )

    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=200)
    for text in out_texts:
        print(text)
        print("-" * 50)
