<div align="center">

<samp>

<h1> BoundaryNet </h1>

<h2> An Attentive Deep Network with Fast Marching Distance Maps for Semi-automatic Layout Annotation </h2>

</samp>


**_To appear at [ICDAR 2021](https://icdar2021.org/)_**

| **[ [```Paper```](<>) ]** | **[ [```Website```](<https://ihdia.iiit.ac.in/BoundaryNet/>) ]** |
|:-------------------:|:-------------------:|

<br>

<img src="Architecture.png">

We propose a novel resizing-free approach for high-precision semi-automatic layout annotation. The variable-sized user selected region
of interest is first processed by an attention-guided skip network. The network optimization is guided via Fast Marching distance maps to obtain a good quality
initial boundary estimate and an associated feature representation. These outputs are processed by a Residual Graph Convolution Network optimized using Hausdorff
loss to obtain the final region boundary.

---

</div>


# Dependencies and Installation

The PALMIRA code is tested with

- Python (`3.5.x`)
- PyTorch (`1.0.0`)
- CUDA (`10.2`)

Please install dependencies by

```bash
pip install -r requirements.txt
```

# Usage

```cd CODE```

## Initial Setup:

- Download the Indiscapes **[[`Dataset Link`](https://github.com/ihdia/indiscapes)]**
- Place the
    - Dataset Images under `data` directory
    - Pretrained BNet Model weights in the `checkpoints` directory
    - JSON annotation data in `datasets` directory


## Training

1. MCNN:
```
bash Scripts/train_mcnn.sh
```
2. Anchor GCN:
```
bash Scripts/train_agcn.sh
```
3. End-to-end Fine Tuning:
```
bash Scripts/fine_tune.sh
```
- For all of the above scripts, corresponding experiment files are present in ```experiments/``` directory.
- Any required parameter changes can be performed in these files.


## Inference

To perform inference and get quantitative results on the test set.

```
bash Scripts/test.sh 
```
> Check the qualitative results at ```visualizations/test_gcn_pred/```.

## Custom Images

- Add Document-Image path and Bounding Box coordinates in ```experiments/test_instance.json``` file.
- Execute -
```bash
python test_instance.py --exp experiments/test_instance.json
```
> Check the corresponding instance-level boundary results at ```visualizations/test_single_img/```.


# Citation

If you use BoundaryNet, please use the following BibTeX entry.

```bibtex
@inproceedings{trivedi2021boundarynet,
    title = {BoundaryNet: An Attentive Deep Network with Fast Marching Distance Maps for Semi-automatic Layout Annotation},
    author = {Trivedi, Abhishek and Sarvadevabhatla, Ravi Kiran},
    booktitle = {International Conference on Document Analysis Recognition, {ICDAR} 2021},
    year = {2021},
}
```

# Contact

For any queries, please contact [Dr. Ravi Kiran Sarvadevabhatla](mailto:ravi.kiran@iiit.ac.in.)

# License

This project is open sourced under [MIT License](LICENSE).
