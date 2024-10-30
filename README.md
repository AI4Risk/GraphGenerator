# DeepGraphGenerator

Awesome Deep Graph Generator

Source codes implementation of papers:

- `GraphRNN`: GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models, in ICML 2018.



## Usage

### Data processing

1. Run `python experiment/preprocess.py` to process a graph into a sparse matrix and save the matrix as an `.npz` file.
2. You can run `python experiment/eval.py` to review the information about the datasets.



### Training

To test implementations of `GraphRNN` and `GraphRNN-S`, run

```
python main.py --method GraphRNN
python main.py --method GraphRNN-S
```

Configuration files can be found in `config/GraphRNN.yaml`.



### Evalutaion

TODO.

### Data Description

The following datasets are from [linqs](https://linqs.org/datasets/) and [snap](https://snap.stanford.edu/data/). More datasets will be added, especially financial graph data, such as `YelpChi`.

| Data     | #Nodes | #Edges  | $d_{mean}$ | GINI  | PWE   |
| -------- | ------ | ------- | ---------- | ----- | ----- |
| Citeseer | 3327   | 4732    | 2.774      | 0.435 | 2.420 |
| Cora     | 2708   | 5429    | 3.898      | 0.405 | 1.932 |
| Pubmed   | 19717  | 44338   | 4.496      | 0.604 | 2.176 |
| Google   | 875713 | 4322051 | 9.871      | 0.587 | 1.617 |

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
```



### Contributors :

<a href="https://github.com/AI4Risk/GraphGenerator/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Risk/GraphGenerator" /></a>



### Citing

If you find *GraphGenerator* is useful for your research, please consider citing the following papers:

```
@InProceedings{pmlr-v80-you18a,
  title={GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models},
  author={You, Jiaxuan and Ying, Rex and Ren, Xiang and Hamilton, William and Leskovec, Jure},
  booktitle={Proceedings of the 35th International Conference on Machine Learning},
  year={2018}
}
```
