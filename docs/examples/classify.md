# Text classification with LoRA

`ferrotensor classify-train` fine-tunes a Llama-style decoder as a text
classifier: LoRA adapters on the decoder, plus a small head over the hidden
states of chosen blocks. `ferrotensor classify-predict` writes the class
probabilities of every line of a file. The library side is
`ferrotensor_trainer::ClassifierTrainer` and the
`ferrotensor_trainer::classify` module.

## How the classifier works

For every tapped block (`--taps`, 1-based; tap `l` is the output of block
`l`), the hidden states are averaged over each text's real tokens, normalised
with RMSNorm and mapped to one logit per label by a bias-free linear layer.
The logits of all taps are summed. By default the first token of every text
is left out of the average: in a causal decoder the first position sees only
itself and often acts as an attention sink with very large activations.
`--no-skip-first-token` keeps it.

The model is built only up to the deepest tap, so blocks after it are never
uploaded or run. With `--taps 12` on a 30-block model only blocks 1 to 12
exist on the GPU, and the adapter file holds only their LoRA tensors.

Training updates the LoRA tensors with AdamW at `--lr` and the head with a
second AdamW at `--head-lr` (weight decay 0.01 for both). Every
`--grad-accum` batches the gradients are averaged, the global norm over all
trainable tensors is clipped to `--clip-grad-norm`, and both optimizers step.
Batches left at the end of the run that do not fill a window are averaged
and applied as one last step.
Both rates follow the same schedule: linear warmup over `--warmup-steps`,
then cosine decay to a tenth of the peak at the last step. A step whose
gradient norm is not finite is skipped and logged as skipped.

## Data

One JSON object per line:

```json
{"id": "a-17", "text": "The first passage ...", "label": "model-a"}
{"id": "b-03", "text": "Another passage ...", "label": "model-b"}
```

| Field | Type | |
|---|---|---|
| `text` | string | required |
| `label` | string | required for `classify-train` and `--eval`; optional for `classify-predict` |
| `id` | string or number | optional; defaults to the 0-based line number |

Other fields are ignored. Row `i` is line `i`, so empty lines are an error
(except at the end of the file). Labels are sorted in byte order and class
`i` is the `i`-th label; the list is saved in `classifier.json`. Texts are
tokenized with the model's tokenizer without special tokens and without a
chat template, and cut to `--max-len` tokens (default 384).

## Train

```bash
ferrotensor classify-train \
  --model models/SmolLM2-360M \
  --data train.jsonl --eval val.jsonl \
  --output runs/cls \
  --taps 16,24 --epochs 2 --batch-size 8 --grad-accum 2
```

| Option | Default | |
|---|---|---|
| `--taps` | the last block | comma-separated block numbers |
| `--max-len` | 384 | tokens per text |
| `--rank`, `--alpha` | 16, 32 | LoRA rank and alpha |
| `--lora-targets` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | projections with adapters |
| `--lr`, `--head-lr` | 1e-4, 1e-3 | peak learning rates |
| `--epochs`, `--batch-size` | 2, 8 | ignored with `--batches` |
| `--grad-accum` | 2 | batches per optimizer step |
| `--clip-grad-norm` | 1.0 | 0 turns clipping off |
| `--warmup-steps` | 0 | |
| `--seed` | 0 | head initialisation and data order |
| `--no-skip-first-token` | off | pool over the first token too |
| `--batches` | none | fixed batch plan, see below |
| `--eval` | none | labelled JSONL scored after every epoch |
| `--log-every` | 10 | console table every N steps |

`--device`, `--f32-weights` work as for `train` (frozen base weights are f16
by default).

Without `--batches`, each epoch shuffles the rows with a generator seeded by
`--seed` and the epoch number, sorts each chunk of 50 batches by token count
(so a batch holds texts of similar length and little padding) and shuffles
the order of the batches. The same seed gives the same order.

With `--batches FILE`, the file sets every batch of the run, in order: one
JSON array of 0-based row indices into `--data` per line.

```json
[0, 4, 9, 12]
[1, 2, 3, 30]
```

`--epochs`, `--batch-size` and shuffling are then ignored; an index outside
the data or an empty batch is an error. This lets a script decide which texts
share a batch.

After every epoch, `--eval` prints the loss, the accuracy and the balanced
accuracy (the mean recall over the labels that occur in the file), and the
output directory is rewritten. It holds:

| File | Contents |
|---|---|
| `adapter.safetensors` | LoRA tensors, in the same format as `train` writes |
| `head.safetensors` | head tensors `classifier.norms.{tap}.weight` `[hidden]` and `classifier.heads.{tap}.weight` `[hidden, labels]` |
| `classifier.json` | `labels` (in class order), `taps`, `max_len`, `skip_first_token`, `rank`, `alpha`, `lora_targets`, `base_model`, `seed`, `ferrotensor_version` |
| `train_log.jsonl` | one line per optimizer step: `step`, `epoch`, `loss` (mean over the step's batches), `grad_norm` (before clipping), `lr`, `head_lr`, `seconds`, `skipped` |

## Predict

```bash
ferrotensor classify-predict \
  --model models/SmolLM2-360M \
  --classifier runs/cls \
  --data test.jsonl --output probs.jsonl
```

Each input line gives one output line, in the same order:

```json
{"id":"a-17","probs":[0.91,0.06,0.03],"label":"model-a"}
```

`probs` are in class order (the `labels` of `classifier.json`) and sum to 1;
`label` is the most likely one. Taps, `max_len` and first-token pooling come
from `classifier.json`. When the input lines have labels, the accuracy and
balanced accuracy over them are printed. `--batch-size` (default 8) sets how
many texts go through the model together; padding does not change the
probabilities.
