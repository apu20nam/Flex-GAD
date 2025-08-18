# MRF-GAD



## Abstract

Detecting anomalous nodes in attributed networks, where each
node is associated with both structural connections and descriptive
attributes, is essential for identifying fraud, misinformation, and
suspicious behavior in domains such as social networks, academic
citation graphs, and e-commerce platforms. We propose Flex-GAD,
a novel unsupervised framework for graph anomaly detection at the
node level. Flex-GAD integrates two encoders to capture comple-
mentary aspects of graph data. The framework incorporates a novel
community-based GCN encoder to model intra-community and
inter-community information into node embeddings, thereby ensur-
ing structural consistency, along with a standard attribute encoder.
These diverse representations are fused using a self-attention-based
representation fusion module, which enables adaptive weighting
and effective integration of the encoded information. This fusion
mechanism allows automatic emphasis of the most relevant node
representation across different encoders. We evaluate Flex-GAD
on seven real-world attributed graphs with varying sizes, node de-
grees, and attribute homogeneity. Flex-GAD achieves an average
AUC improvement of 7.98% over the previously best-performing
method, GAD-NR, demonstrating its effectiveness and flexibility
across diverse graph structures. Moreover, it significantly reduces
training time, running 102× faster per epoch than Anomaly
DAE and 3× faster per epoch than GAD-NR on average across
seven benchmark datasets.

## Architecture of the Flex-GAD

<img width="2521" height="875" alt="Flex GAD Diagram (6)" src="https://github.com/user-attachments/assets/d9589291-6551-49a4-974a-e7b211d01fab" />



## Installation
### Requirements 
```
Python >= 3.10
cuda == 12.2
Pytorch==1.5.1
PyG: torch-geometric==1.5.0
```
#### If you have above configuration you can directly use this given .yml file to create conda  envirnment 
```
conda env create -f environment.yml
```
#### To run the code 
```
python run.py --dataset --lr --lambda_loss1 --lambda_loss2 --dimension --num_epochs
```
