# Hardware Originally Used
The training of this model was done on two machines thanks to preprocessing some of the audio.

All training was done on Ubuntu Server 24.04 LTS and CUDA 13.0.

Machine 1:
CPU: Ryzen Threadripper 2970WX 24-Core Processor
GPU: 3 x rtx 3090 24gb, 1 x rtx 3090ti 24gb
Memory: 128gb DDR4 3600MHz ECC

Machine 2:
CPU: Ryzen 7 5800x 8-core Processor
GPU: 1 x rtx 5070 12gb
Memory: 64gb DDR4 3600MHz Non-ECC
# Requirements
- `ffmpeg` (6->8)
- `uv`
- `python` (3.13)
- `cuda toolkit` (13.0)
- `hf` (Hugging Face)

CUDA is required for this project as many optimizations rely on it.

## Step 1:
```bash
uv sync
source .venv/bin/activate
```
## Step 2:
If you haven't already install `hf`:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```
Read the `README.md` in datasets to install the required datasets.

## Step 3:
Download the necessary models into cache.
```bash
hf download TuKoResearch/AuriStream100M_RoPE_librilight
hf download google/embeddinggemma-300m
```

Now you should be ready to continue.

# Following the stages I went through to train the model:
## Stage 1 - Machine 1/2
___
This stage was manual data assortment and formatting, and precomputation. All details for this stage are included in the `README.md` in the datasets directory.

Due to fast iterative development, precomputing the roughly ~1.2 million audio files in LAION's got talent enhanced was absolutely necessary.

## Stage 2 - Machine 2
___
This stage was the CLAP linear layer projector training step. Due to the embeddings precomputed in Stage 1 this stage would benefit from faster inference speed over vram space, so it was trained on Machine 2.

The projector heads were first trained with:
```bash
uv run torchrun --standalone --nproc_per_node=1 clap_train.py
```
And then had it's projectors extracted via:
```bash
uv run export_projectors.py
```
The resulting `.pt` file was placed in the models directory.
## Stage 3 - Machine 2
___
The training for this stage was done iteratively with checkpoints to avoid overfitting the commands chained as followed:

1. 50 epochs
```bash
uv run torchrun --nproc_per_node=1 train_slm.py \
  --num-emotions 36 \
  --data-root datasets/LAION \
  --clap-ckpt checkpoints/clap_projectors.pt \
  --batch-size 512 \
  --epochs 50
```
2. 100 epochs (renamed to `slm_stage3_epoch49_with_splits.pt` after testing different options)
```bash
uv run torchrun --nproc_per_node=1 train_slm.py \
  --num-emotions 36 \
  --data-root datasets/LAION \
  --clap-ckpt checkpoints/clap_projectors.pt \
  --batch-size 512 \
  --epochs 100 \
  --checkpoint checkpoints/slm_stage3_epoch049_with_splits.pt \
  --resume
```
3. 150 epochs (renamed to slm_stage3_epoch099_best.pt)
```bash
uv run torchrun --nproc_per_node=1 train_slm.py \
  --num-emotions 36 \
  --data-root datasets/LAION \
  --clap-ckpt checkpoints/clap_projectors.pt \
  --batch-size 512 \
  --epochs 150 \
  --checkpoint checkpoints/slm_stage3_epoch099_best.pt \
  --resume
```
## Stage 4 - Machine 1
___
Due to not precomputing embeddings, MELD had to have plenty of VRAM to run fast, training time was around 30 minutes on average for me.

This step was done by fine tuning the model for the MELD dataset using the training and eval split and then subsequently evaluating the performance of the model on the test split.

The command used was:
```bash
uv run python meld_train_slm.py \
  --data-root datasets/MELD \
  --stage3-ckpt models/slm_stage3_epoch149_best.pt \
  --clap-ckpt models/clap_projectors.pt \
  --mode A \
  --epochs 60 \
  --batch-size 128 \
  --lr 0.001 \
  --weight-decay 0.01 \
  --num-workers 4 \
  --save-dir checkpointA_balanced \
  --resume checkpointA_balanced/meld_slm_A_best.pt \
  --seed 17
```

To evaluate I prepared 3 scripts (these will not run without the datasets present):
- `eval_stage2.py`
- `eval_stage3.py`
- `eval_stage4.py`
Using `uv run ...` will run them and produce their metrics.
