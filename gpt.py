"""A small character-level GPT, implemented with explicit PyTorch building blocks."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        head_size = key.size(-1)
        weights = query @ key.transpose(-2, -1) * head_size**-0.5
        weights = weights.masked_fill(self.mask[:steps, :steps] == 0, float("-inf"))
        weights = self.dropout(F.softmax(weights, dim=-1))
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(torch.cat([head(x) for head in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Block(nn.Module):
    """Pre-layer-norm Transformer block with residual connections."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.feed_forward(self.ln2(x))


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.language_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, steps = idx.shape
        if steps > self.block_size:
            raise ValueError(f"sequence length {steps} exceeds block size {self.block_size}")
        positions = torch.arange(steps, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        logits = self.language_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    if len(data) <= block_size:
        raise ValueError("corpus is too short for the chosen block size")
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    model.eval()
    results = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(args.eval_batches)
        for i in range(args.eval_batches):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        results[name] = losses.mean().item()
    model.train()
    return results


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    text = Path(args.text).read_text(encoding="utf-8")
    if not text:
        raise ValueError("training text is empty")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = max(1, int(len(encoded) * (1 - args.val_fraction)))
    train_data, val_data = encoded[:split], encoded[split:]
    if len(train_data) <= args.block_size or len(val_data) <= args.block_size:
        raise ValueError("corpus needs more characters; reduce --block-size or --val-fraction")

    config = dict(vocab_size=len(chars), block_size=args.block_size, n_embd=args.n_embd,
                  n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout)
    model = GPTLanguageModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"device={device}  parameters={sum(p.numel() for p in model.parameters()):,}  "
          f"vocabulary={len(chars)}  corpus={len(encoded):,} characters")

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, args, device)
            print(f"step {step:>6} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        if step == args.steps:
            break
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "itos": chars, "model": model.state_dict()}, output)
    print(f"saved checkpoint: {output.resolve()}")


def generate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    try:
        checkpoint: dict[str, Any] = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        checkpoint = torch.load(args.checkpoint, map_location=device)
    chars = checkpoint["itos"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        raise ValueError(f"prompt contains characters absent from training vocabulary: {unknown!r}")
    model = GPTLanguageModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    prompt = args.prompt or chars[0]
    context = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
    output = model.generate(context, args.tokens, args.temperature, args.top_k)
    print("".join(chars[i] for i in output[0].tolist()))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Train and sample a character-level GPT from scratch.")
    commands = root.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train", help="train a model on a UTF-8 text file")
    t.add_argument("--text", default="data/tiny_corpus.txt")
    t.add_argument("--out", default="runs/gpt.pt")
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--eval-interval", type=int, default=150)
    t.add_argument("--eval-batches", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--block-size", type=int, default=64)
    t.add_argument("--n-embd", type=int, default=64)
    t.add_argument("--n-head", type=int, default=4)
    t.add_argument("--n-layer", type=int, default=3)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--learning-rate", type=float, default=3e-4)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--val-fraction", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--cpu", action="store_true", help="force CPU even when CUDA is available")
    t.set_defaults(func=train)
    g = commands.add_parser("generate", help="continue text from a trained checkpoint")
    g.add_argument("--checkpoint", default="runs/gpt.pt")
    g.add_argument("--prompt", default="Once upon a time")
    g.add_argument("--tokens", type=int, default=300)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--cpu", action="store_true")
    g.set_defaults(func=generate)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
