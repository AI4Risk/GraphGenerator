# DeepGraphGenerator

Awesome Deep Graph Generator

Source codes implementation of papers:

- `BTGAE`: **Divide and Conquer: A Topological Heterogeneity-based Framework for Scalable and Realistic Graph Generation**. *(Official PyTorch Implementation)*
- `CPGAE`: **Efficient Learning-based Community-Preserving Graph Generation**, in *ICDE* 2022. (GAE version of CPGAN)

Implementation of baselines:

- `GraphRNN`: **GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models**, in *ICML* 2018.

## Usage

### Data processing

1. Run `python experiment/preprocess.py` to process a graph into a sparse matrix and save the matrix as an `.npz` file.
2. You can run `python experiment/eval.py` to review the information about the datasets.

### Training

To test implementations of the methods, run

```bash
python main.py --method BTGAE
python main.py --method CPGAE
python main.py --method GraphRNN
python main.py --method GraphRNN-S
```

Configuration files can be found in `config/`.

### Evalutaion

The evaluation tools are found in `experiment/eval_tools`. Utilize the functions within `stats1graph.py` for graph statistics and those in `stats2graphs.py` for calculating MMD between ground truth and generated graphs.

### Data Description

The following datasets are mainly from [linqs](https://linqs.org/datasets/) and [snap](https://snap.stanford.edu/data/).

| Data     | #Nodes | #Edges  | $d_{mean}$ | GINI  | PWE   |
| -------- | ------ | ------- | ---------- | ----- | ----- |
| citeseer | 3327   | 4732    | 2.774      | 0.435 | 2.420 |
| cora     | 2708   | 5429    | 3.898      | 0.405 | 1.932 |
| pubmed   | 19717  | 44338   | 4.496      | 0.604 | 2.176 |
| Epinions | 75879  | 508837  | 10.694     | 0.805 | 2.026 |
| google   | 875713 | 5105039 | 9.871      | 0.587 | 1.617 |
| YelpChi  | 45954  | 3846979 | 167.427    | 0.322 | 1.205 |

$d_{mean}$: mean degree.

`GINI`:  GINI index, which is a common measure for inequality in a degree distribution.

`PWE`: power-law exponent.

## Test Result

The performance of models tested on datasets are listed as follows:

TODO.

## Repo Structure

The repository is organized as follows:

- `main.py`: organize all models.
- `experiment/`: preprocessing and evaluation.
- `methods/`: implementations of models.
- `config/`: configuration files for different models.
- `models/`: the checkpoints or the trained models for each method.
- `data/`: dataset files.
- `requirements.txt`: package dependencies.

## Requirements

```
torch           2.3.1+cu121
networkx        2.8
scipy           1.14.1
scikit-learn    1.5.2
numpy           1.26.4
community       1.0.0b1
python-louvain  0.16
dgl             2.4.0+cu121
```

### Contributors :

<a href="https://github.com/AI4Risk/GraphGenerator/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Risk/GraphGenerator" /></a>

### Citing

If you find *GraphGenerator* is useful for your research, please consider citing the following papers:

```bibtex
@inproceedings{xiang2022efficient,
  title={Efficient learning-based community-preserving graph generation},
  author={Xiang, Sheng and Cheng, Dawei and Zhang, Jianfu and Ma, Zhenwei and Wang, Xiaoyang and Zhang, Ying},
  booktitle={2022 IEEE 38th International Conference on Data Engineering (ICDE)},
  pages={1982--1994},
  year={2022},
  organization={IEEE}
}
```