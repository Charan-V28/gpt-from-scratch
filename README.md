# Build GPT From Scratch

A compact, readable GPT implementation that trains a decoder-only Transformer on your own text. The code makes the important pieces explicit: token IDs, learned token and position embeddings, masked multi-head self-attention, feed-forward blocks, residual connections, next-token loss, and autoregressive sampling.

> This is an educational, character-level model. It is not a replica of a production GPT and will not have useful general knowledge from the tiny demo corpus.

## Quick start

Requires Python 3.10+ and PyTorch 2.0+.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python gpt.py train --text data/tiny_corpus.txt --steps 1200 --out runs/demo.pt
python gpt.py generate --checkpoint runs/demo.pt --prompt "Once upon a time" --tokens 300
```

On macOS/Linux, activate with `source .venv/bin/activate`. CUDA is used automatically when available; otherwise training runs on CPU. For a quick smoke run, use `--steps 50`.

## What the model does

```text
text → character IDs → token + position embeddings
     → [causal self-attention → MLP] × N
     → vocabulary logits → next-character loss
```

Every training example is split into an input window `x` and the same window shifted one character to make its target `y`. The triangular attention mask prevents a position from seeing future characters. At generation time, the model repeatedly predicts one next character and appends it to the context.

## Use your own text

Pass a plain UTF-8 text file to `train`. The vocabulary is built from that file, so generation can only emit characters present in the training corpus. Keep the file large enough to provide varied examples; a few megabytes is a reasonable first experiment. Training saves the vocabulary and model configuration inside the checkpoint, so generation does not need the original text file.

```powershell
python gpt.py train --text .\my_corpus.txt --steps 5000 --block-size 128 --batch-size 32 --out runs/my_model.pt
python gpt.py generate --checkpoint runs/my_model.pt --prompt "Your starting text" --tokens 500 --temperature 0.8 --top-k 40
```

## Useful options

`train --help` and `generate --help` list every option. Training supports a validation split, periodic loss reports, gradient clipping, seeded runs, and configurable model size. Smaller `--block-size`, `--batch-size`, `--n-embd`, or `--n-layer` values use less memory. `--top-k` restricts sampling to the most likely next characters; lower temperature makes output more predictable.

## Project map

- `gpt.py` — complete implementation and command-line interface.
- `data/tiny_corpus.txt` — tiny starter corpus for an end-to-end demo.
- `requirements.txt` — PyTorch dependency.

## Limitations

Character tokenization is intentionally simple and inefficient compared with subword tokenizers. A small corpus and a model this size are for learning and experimentation, not for building a capable assistant. The script trains from scratch; it does not download data, call a hosted model, or use pretrained weights.

## Architecture notes

The attention operation is implemented directly with tensor products and a lower-triangular causal mask rather than a black-box Transformer layer. Each attention head computes scaled query-key scores, masks future positions, applies softmax, and mixes value vectors. The feed-forward network expands the channel width by four, applies GELU, and projects back. Layer normalization is applied before attention and before the MLP (pre-norm), with residual additions around both sublayers.
