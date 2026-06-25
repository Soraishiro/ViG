# image captioning

An image captioning framework built on [GRIT](https://arxiv.org/abs/2207.09666) with two additional modules: **Residual Relation Memory (RRM)** for spatial-relational reasoning, and **Local Cultural Query Memory (LCQM)** for culturally-grounded concept learning.

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
│ ├── default_config.yaml  
│ └── custom_config.yaml  
├── tools/  
│ ├── mine_local_phrases.py  
│ ├── build_phrase_supervision.py
│ └── adapt_vocab_format.py  
├── utils/  
├── data/  
├── train.py  
├── dataset.py  
├── training_regimes.py  
├── eval_caption.py  
├── inference_caption.py  
└── eval.py
```

## Setup

### Requirements

- Python >= 3.10, CUDA >= 11.8
- PyTorch >= 2.0, torchvision
- Linux (CUDA ops require GCC + NVCC)

### Install

```shell
git clone https://github.com/Soraishiro/Image_Captioning.git
cd Image_Captioning

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

The pretrained models train, evaluate and inference on [Ktvic: A vietnamese image captioning dataset on the life domain](https://doi.org/10.48550/arXiv.2401.08100) dataset. Available pretrained checkpoints:

| Model                | Link                   |
| -------------------- | ---------------------- |
| full model           | [`ckpts/full_model.pth`](https://drive.google.com/file/d/1-Xi6HZPCJ28Yeaii1SVwHfX-r9OwHO7-/view?usp=sharing) |
| model with RRM only  | [`ckpts/rrm.pth`](https://drive.google.com/file/d/1FTUgGSUIeSpFwO-znj7ZuHsfAYkCiHJB/view?usp=sharing)        |
| model with LCQM only | [`ckpts/lcqm.pth`](https://drive.google.com/file/d/1Gh8syW_XuoXKk9U-MTl_CAgrqxG2EuVp/view?usp=sharing)       |

Set the path to use:

```shell
export CHECKPOINT=ckpts/full_model.pth
```

### Data Preparation

Download and extract the image captioning dataset including train and test images with annotations. Expected directory structure:

```
path/to/dataset/
├── train-images/         # training images
├── public-test-images/   # test images
├── train_data.json       # training annotations
└── test_data.json        # test annotations
```

Set the path via environment variable:

```shell
export DATA_ROOT=/path/to/dataset
```

### Training

Enable components individually or combined:

```shell
# baseline (no additional modules)
python train.py model_ext.rrm.enabled=false model_ext.lcqm.enabled=false

# RRM only
python train.py \
    model_ext.rrm.enabled=true \
    optimizer.freeze_grit=true \

# LCQM only (requires phrase data — see Phrase Mining below)
python train.py \
    model_ext.lcqm.enabled=true \
    optimizer.freeze_grit=true \

# Full (RRM + LCQM with stop-grad connection)
python train.py \
    model_ext.rrm.enabled=true \
    model_ext.lcqm.enabled=true \
    model_ext.lcqm.use_rel_input=true \
    model_ext.lcqm.stop_grad_rel=true \
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
python eval.py exp.checkpoint=path/to/checkpoint.pth
```

## Inference

```shell
python inference_caption.py \
    +img_path=path/to/image.jpg \
    +vocab_path=data/vocab.json \
    exp.checkpoint=path/to/checkpoint.pth
```

## Acknowledgement

This work builds upon [GRIT](https://github.com/davidnvq/grit), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and [M2-Transformer](https://github.com/aimagelab/meshed-memory-transformer). We sincerely thank the authors of these open source projects.
