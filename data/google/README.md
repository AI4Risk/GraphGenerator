# Google web graph

### Dataset information

Nodes represent web pages and directed edges represent hyperlinks between them. The data was released in 2002 by Google as a part of [Google Programming Contest](http://www.google.com/programming-contest/).

| Dataset statistics               |                 |
| :------------------------------- | --------------- |
| Nodes                            | 875713          |
| Edges                            | 5105039         |
| Nodes in largest WCC             | 855802 (0.977)  |
| Edges in largest WCC             | 5066842 (0.993) |
| Nodes in largest SCC             | 434818 (0.497)  |
| Edges in largest SCC             | 3419124 (0.670) |
| Average clustering coefficient   | 0.5143          |
| Number of triangles              | 13391903        |
| Fraction of closed triangles     | 0.01911         |
| Diameter (longest shortest path) | 21              |
| 90-percentile effective diameter | 8.1             |

### Source (citation)

- J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney. [Community Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters](http://arxiv.org/abs/0810.1355). Internet Mathematics 6(1) 29--123, 2009.
- Google programming contest, 2002

### Files

| File                                                         | Description                                        |
| :----------------------------------------------------------- | :------------------------------------------------- |
| [web-Google.txt.gz](https://snap.stanford.edu/data/web-Google.txt.gz) | Webgraph from the Google programming contest, 2002 |