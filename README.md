# ViG: Grounded Adaptation of Transformer-Based Models for Low-Resource Vietnamese Image Captioning

Implementation of **ViG** — a dual-enhancement framework adapting [GRIT](https://arxiv.org/abs/2207.09666) for Vietnamese image captioning. ViG inserts two lightweight modules into the frozen GRIT visual stack: **Residual Relation Memory (RRM)** for spatial-relational reasoning, and **Local Cultural Query Memory (LCQM)** for culturally-grounded concept learning. This serves as the official implementation for undergraduate thesis (CS505) at the University of Information Technology - VNU-HCM.

## Project Structure

```md
├── models/
│ ├── caption/
│ │ ├── transformer.py  
│ │ ├── relation_memory.py  
│ │ ├── cultural_memory.py  
│ │ ├── cap_generator.py  
│ │ ├── detector.py  
│ │ ├── grid_net.py  
│ │ ├── base.py  
│ │ └── containers.py  
│ ├── common/  
│ ├── detection/  
│ └── ops/  
├── engine/  
├── datasets/caption/  
├── configs/caption/
│ ├── vicap_config.yaml  
│ └── custom_config.yaml  
├── tools/  
│ ├── mine_local_phrases.py  
│ ├── build_phrase_supervision.py
│ └── adapt_vocab_format.py  
├── utils/  
├── data/  
├── train_vicap.py  
├── vicap_dataset.py  
├── training_regimes.py  
├── eval_caption.py  
├── inference_caption.py  
└── official_test_vicap.py
```

## Setup

### Requirements

- Python >= 3.10, CUDA >= 11.8
- PyTorch >= 2.0, torchvision
- Linux (CUDA ops require GCC + NVCC)

### Install

```shell
git clone <repo-url>
cd vig

python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Build deformable attention ops
cd models/ops
python setup.py build develop
python test.py
cd ../..
```

### Checkpoints

Download pretrained checkpoints and set environment variables:

| Checkpoint                     | Description                 | Link                  |
| ------------------------------ | --------------------------- | --------------------- |
| ViG full (RRM+LCQM, stop-grad) | Full ViG with stop-gradient | `ckpts/full_vig.pth`  |
| ViG LCQM-only                  | ViG with LCQM only          | `ckpts/lcqm_only.pth` |
| ViG RRM-only                   | ViG with RRM only           | `ckpts/rrm_only.pth`  |

### Training

Enable components individually or combined:

```shell
# GRIT baseline (no ViG modules)
python train_vicap.py model_vig.rrm.enabled=false model_vig.lcqm.enabled=false

# RRM only
python train_vicap.py \
    model_vig.rrm.enabled=true \
    optimizer.freeze_grit=true \

# LCQM only (requires phrase data — see Phrase Mining below)
python train_vicap.py \
    model_vig.lcqm.enabled=true \
    optimizer.freeze_grit=true \

# Full ViG (RRM + LCQM with stop-grad connection)
python train_vicap.py \
    model_vig.rrm.enabled=true \
    model_vig.lcqm.enabled=true \
    model_vig.lcqm.use_rel_input=true \
    model_vig.lcqm.stop_grad_rel=true \
    optimizer.freeze_grit=true \
```

### Phrase Mining (for LCQM)

LCQM requires precomputed phrase supervision data. Generate from your dataset:

```shell
# Step 1: Mine cultural phrases from training captions
python tools/mine_local_phrases.py \
    --caption_json path/to/captions.json \
    --output data/local_phrase_vocab.json

# Step 2: Build phrase embeddings, positives, and token masks
python tools/build_phrase_supervision.py \
    --caption_json path/to/captions.json \
    --phrase_vocab data/local_phrase_vocab.json \
    --output_dir data/

# Output: data/phrase_embeddings.npy
#         data/phrase_list.json
#         data/phrase_positives_train.json
#         data/token_masks_train.json
```

## Evaluation

```shell
export DATA_ROOT=/path/to/dataset
python official_test_vicap.py exp.checkpoint=path/to/checkpoint.pth
```

## Inference

```shell
python inference_caption.py \
    +img_path=path/to/image.jpg \
    +vocab_path=data/vocab.json \
    exp.checkpoint=path/to/checkpoint.pth
```

## Acknowledgement

ViG builds upon [GRIT](https://github.com/davidnvq/grit), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and [M2-Transformer](https://github.com/aimagelab/meshed-memory-transformer). We sincerely thank the authors of these open source projects.
