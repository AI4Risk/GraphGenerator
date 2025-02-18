# DeepGraphGenerator

Awesome Deep Graph Generator

Source codes implementation of papers:

- `BTGAE`: **Divide and Conquer: A Topological Heterogeneity-based Framework for Scalable and Realistic Graph Generation**. *(Official PyTorch Implementation)*
- `VRDAG`: **Efficient Dynamic Attributed Graph Generation**, in *ICDE* 2025.
- `TGAE`: **Efficient Learning-based Graph Simulation for Temporal Graphs**, in *ICDE* 2025.
- `CPGAE`: **Efficient Learning-based Community-Preserving Graph Generation**, in *ICDE* 2022. (GAE version of CPGAN)

Implementation of baselines:

- `GraphRNN`: **GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models**, in *ICML* 2018.

## Usage

### Data processing

1. Run `python experiment/preprocess.py` to process a graph into a sparse matrix and save the matrix as an `.npz` file or `.pkl` file.

### Training

To test implementations of the methods, run

```bash
python main.py --method BTGAE
python main.py --method VRDAG
python main.py --method TGAE
python main.py --method CPGAE
python main.py --method GraphRNN
python main.py --method GraphRNN-S
```

All predefined configurations are in `config/`. Dynamically override parameters using:

```bash
python train.py --method <name> --update <key1>=<value1> <key2>=<value2> ...
```

Key values shall exactly match (case-sensitive) the corresponding parameters defined in configuration files.

Examples:

```bash
python main.py --method BTGAE --update data=cora epochs=150 learning_rate=3e-3

python main.py --method TGAE --update epochs=100 lr=1e-3
```

### Evalutaion

The evaluation tools are located in the `experiment/graph_metrics` directory. For usage instructions, see [README.md](https://github.com/AI4Risk/GraphGenerator/blob/main/experiment/graph_metrics/README.md).

### Data Description

The following datasets are mainly from [linqs](https://linqs.org/datasets/) and [snap](https://snap.stanford.edu/data/).

| Data     | #Nodes  | #Edges    | $d_{mean}$ | GINI  | PWE   |
| -------- | ------- | --------- | ---------- | ----- | ----- |
| citeseer | 3,327   | 4,732     | 2.774      | 0.435 | 2.420 |
| cora     | 2,708   | 5,429     | 3.898      | 0.405 | 1.932 |
| pubmed   | 19,717  | 44,338    | 4.496      | 0.604 | 2.176 |
| Epinions | 75,879  | 508,837   | 10.694     | 0.805 | 2.026 |
| google   | 875,713 | 5,105,039 | 9.871      | 0.587 | 1.617 |
| YelpChi  | 45,954  | 3,846,979 | 167.427    | 0.322 | 1.205 |

$d_{mean}$: mean degree.

`GINI`:  GINI index, which is a common measure for inequality in a degree distribution.

`PWE`: power-law exponent.



The following **temporal network datasets** are from [snap](https://snap.stanford.edu/data/) and [Network Repository](https://networkrepository.com/). For more information, please refer to the [VRDAG paper](https://arxiv.org/abs/2412.08810).

| Data          | #Nodes | #Edges  | T    |
| ------------- | ------ | ------- | ---- |
| Emails-DNC    | 1,891  | 39,264  | 14   |
| Bitcoin-Alpha | 3,783  | 24,186  | 37   |
| Wiki-Vote     | 7,115  | 103,689 | 43   |

## Test Result

The performance of models tested on static graph datasets are listed as follows:

| Dataset  | Method | Deg. dist. | Clus. dist. | Wedge count | Triangle count |  LCC   |  PLE   |  Gini  | Clus. coef. |
| :------: | :----: | :--------: | :---------: | :---------: | :------------: | :----: | :----: | :----: | :---------: |
| Citeseer | CPGAE  |   0.0035   |   0.0124    |   0.0812    |     0.0274     | 0.066  | 0.003  | 0.0026 |   0.1182    |
|          | BTGAE  |   0.0029   |   0.0119    |   0.0402    |     0.335      | 0.0085 | 0.0711 | 0.0208 |   0.2473    |
|   Cora   | CPGAE  |   0.004    |   0.0088    |   0.1333    |     0.056      | 0.0193 | 0.0026 | 0.0252 |   0.2161    |
|          | BTGAE  |   0.0061   |   0.0078    |   0.0026    |     0.054      | 0.0129 | 0.0028 | 0.0113 |   0.0376    |
|  Pubmed  | CPGAE  |   0.0156   |   0.0144    |    0.423    |     0.2835     |   0    | 0.0807 | 0.0994 |   1.2245    |
|          | BTGAE  |   0.0164   |   0.0093    |    0.227    |     0.1541     | 0.0512 | 0.1233 | 0.0737 |   0.0307    |
| Epinions | CPGAE  |   0.0175   |   0.0362    |   0.6739    |     0.3723     |   0    | 0.1768 | 0.1058 |   0.9246    |
|          | BTGAE  |   0.0189   |   0.0265    |   3.3051    |     3.3247     | 0.0538 | 0.1394 | 0.0597 |   0.0104    |
| YelpChi  | CPGAE  |   0.0236   |    0.02     |   0.1487    |     0.133      | 0.0012 | 0.0416 | 0.3062 |   0.0186    |
|          | BTGAE  |   0.0286   |   0.0396    |   0.0361    |     0.6578     | 0.0057 | 0.0037 | 0.1709 |   0.6429    |

The second table shows the test results of dynamic graph generation methods.

|    Dataset    | Method | Deg. dist. | Clus. dist. | Wedge count | Triangle count |   LCC   |  PLE   |  Gini  | Clus. coef. |
| :-----------: | :----: | :--------: | :---------: | :---------: | :------------: | :-----: | :----: | :----: | :---------: |
|  Emails-DNC   | VRDAG  |   0.0084   |   0.0473    |   0.8118    |     0.8705     | 1.5973  | 0.0735 | 0.0837 |   0.2465    |
|               |  TGAE  |   0.0027   |   0.0144    |   0.4768    |     0.4273     | 0.2105  | 0.1353 | 0.0199 |   0.2641    |
| Bitcoin-Alpha | VRDAG  |   0.0052   |   0.0121    |   0.8572    |     0.7422     | 0.8909  | 0.1263 | 0.0652 |   0.9037    |
|               |  TGAE  |   0.0019   |   0.0035    |   0.3014    |     0.3636     | 0.2712  | 0.2009 | 0.0096 |   0.5274    |
|   Wiki-Vote   | VRDAG  |   0.0404   |   0.0656    |   4.2868    |     0.8584     | 12.2631 | 0.1463 | 0.3611 |   0.9466    |
|               |  TGAE  |   0.0037   |   0.0075    |   0.2768    |     0.2843     | 0.1786  | 0.4262 | 0.0024 |   0.1824    |

The evaluation metrics used in these tests are as follows:

- **Deg. dist.:** It measures the degree distribution similarity between the generated graph and the real graph using Maximum Mean Discrepancy (MMD). A smaller MMD value indicates a closer degree distribution. 
- **Clus. dist.:** This metric focuses on the clustering coefficient distribution similarity using MMD.

- **Wedge count**: It counts the number of wedges in the graph.
- **Triangle count**: The number of triangles in the graph is counted.
- **LCC**: It represents the size of the largest connected component in the graph.
- **PLE**: It is the power-law exponent associated with the degree distribution of the graph. 
- **Gini**: GINI index, which is a common measure for inequality in a degree distribution.
- **Clus. coef.**: The global clustering coefficient of the graph.

For temporal graphs, given a metric $f_m(\cdot)$, the real graph $\widetilde{G}$, and the synthetic one $\widetilde{G^{\prime}}$, we construct a sequence of snapshots $\widetilde{S}^t$ ($\widetilde{S^{\prime}}^t$), $t = 1, ...,T$, of $\widetilde{G}$ ($\widetilde{G^{\prime}}$) by aggregating edges from the initial timestamp to the current timestamp $t$. Then, we measure the average difference (in percentage) of the given metric $f_m(\cdot)$ between two graphs as follows:

$$
f_{avg}(\widetilde{G},\widetilde{G^{\prime}},f_m)=Mean_{t=1:T}(|\frac{f_m(\widetilde{S}^t)-f_m(\widetilde{S^{\prime}}^t)}{f_m(\widetilde{S}^t)}|)
$$


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
@inproceedings{li2025efficient,
  title={Efficient Dynamic Attributed Graph Generation},
  author={Li, Fan and Wang, Xiaoyang and Cheng, Dawei and Chen, Cong and Zhang, Ying and Lin, Xuemin},
  booktitle={2025 IEEE 41th International Conference on Data Engineering (ICDE)},
  year={2025},
  organization={IEEE}
}

@inproceedings{xiang2025efficient,
  title={Efficient Learning-based Graph Simulation for Temporal Graphs},
  author={Xiang, Sheng and Xu, Chenhao and  Cheng, Dawei and Wang, Xiaoyang and Zhang, Ying},
  booktitle={2025 IEEE 41th International Conference on Data Engineering (ICDE)},
  year={2025},
  organization={IEEE}
}

@inproceedings{xiang2022efficient,
  title={Efficient learning-based community-preserving graph generation},
  author={Xiang, Sheng and Cheng, Dawei and Zhang, Jianfu and Ma, Zhenwei and Wang, Xiaoyang and Zhang, Ying},
  booktitle={2022 IEEE 38th International Conference on Data Engineering (ICDE)},
  pages={1982--1994},
  year={2022},
  organization={IEEE}
}

@article{xiang2022general,
  title={General graph generators: experiments, analyses, and improvements},
  author={Xiang, Sheng and Wen, Dong and Cheng, Dawei and Zhang, Ying and Qin, Lu and Qian, Zhengping and Lin, Xuemin},
  journal={The VLDB Journal},
  pages={1--29},
  year={2022},
  publisher={Springer}
}
```