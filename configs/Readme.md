<div align="center">

<samp>

<h1> BoundaryNet -- Baselines</h1>

</samp>

</div>


# CurveGCN

The code is tested with

- Python (`3.5.x`)
- PyTorch (`1.0.0`)
- CUDA (`10.2`)


```cd baselines/CureveGCN/```


Please install dependencies by

```bash
pip install -r requirements.txt
```

## Usage


### Initial Setup:

- Download the Indiscapes dataset - **[[`Dataset Link`](https://github.com/ihdia/indiscapes)]**
- Place the
    - Dataset Images under `doc_images` directory
    - Pretrained Model weights in the `checkpoints` directory
    - JSON annotation data in `data_splits` directory


### Training & Inference

```
python3 combined_train.py --exp train_experiment.json
```

- Any required parameter changes can be performed in the experiment file.



# PolyRNN ++ 

The code is tested with

- Python (`3.5.x`)
- PyTorch (`0.4.0`)
- CUDA (`10.2`)


```cd baselines/polyrnn-pp-pytorch/code/```


Please install dependencies by

```bash
pip install -r code/requirements.txt
```

## Usage


### Initial Setup:

- Download the Indiscapes dataset - **[[`Dataset Link`](https://github.com/ihdia/indiscapes)]**
- Place the
    - Dataset Images under `doc_images` directory
    - Pretrained CurveGCN/DACN Model weights in the `checkpoints` directory
    - JSON annotation data in `data_splits` directory


### Training & Inference

```
python3 Scripts/train/train_ce.py --exp Experiments/mle.json
```

- Any required parameter changes can be performed in the experiment file.



# DACN

The code is tested with

- Python (`3.5.x`)
- PyTorch (`1.0.0`)
- CUDA (`10.2`)


```cd baselines/DACN/```


Please install dependencies by

```bash
pip install -r requirements.txt
```

## Usage


### Initial Setup:

- Download the Indiscapes dataset - **[[`Dataset Link`](https://github.com/ihdia/indiscapes)]**
- Place the
    - Dataset Images under `doc_images` directory
    - Pretrained Model weights in the `checkpoints` directory
    - JSON annotation data in `data_splits` directory


### Training & Inference

```
python3 combined_train.py --exp train_experiment.json
```

- Any required parameter changes can be performed in the experiment file.
