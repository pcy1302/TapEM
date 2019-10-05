# Task-Guided Pair Embedding in Heterogeneous Network (TapEm)

<img src="https://github.com/pcy1302/TapEM/blob/master/motivations.png" height="300">

### Overview
> Many real-world tasks solved by heterogeneous network embedding methods can be cast as modeling the likelihood of a pairwise relationship between two nodes. For example, the goal of author identification task is to model the likelihood of a paper being written by an author (paper–author pairwise relationship). Existing taskguided embedding methods are node-centric in that they simply measure the similarity between the node embeddings to compute the likelihood of a pairwise relationship between two nodes. However, we claim that for task-guided embeddings, it is crucial to focus on directly modeling the pairwise relationship. In this paper, we propose a novel task-guided pair embedding framework in heterogeneous network, called TaPEm, that directly models the relationship between a pair of nodes that are related to a specific task (e.g., paper-author relationship in author identification). To this end, we 1) propose to learn a pair embedding under the guidance of its associated context path, i.e., a sequence of nodes between the pair, and 2) devise the pair validity classifier to distinguish whether the pair is valid with respect to the specific task at hand. By introducing pair embeddings that capture the semantics behind the pairwise relationships, we are able to learn the fine-grained pairwise relationship between two nodes, which is paramount for task-guided embedding methods. Extensive experiments on author identification task demonstrate that TaPEm outperforms the state-of-the-art methods, especially for authors with few publication records.

<img src="https://github.com/pcy1302/TapEM/blob/master/model.png" height="500">

### Paper
- [Task-Guided Pair Embedding in Heterogeneous Network](https://arxiv.org/pdf/1906.01546.pdf) (*CIKM 2019*)
  - [_**Chanyoung Park**_](http://pcy1302.github.io), Donghyun Kim, Qi Zhu, Jiawei Han, Hwanjo Yu

### Requirements

- Python version: 3.6.8
- Pytorch version: 1.2.0
  

### How to Run

```
git clone https://github.com/pcy1302/TapEM.git
cd TapEM
cd code
python main.py --embedder TapEM --pretrain
```
