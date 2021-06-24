# `PolyGen PyTorch`

A PyTorch implementation of the [PolyGen: An Autoregressive Generative Model of 3D Meshes](https://arxiv.org/abs/2002.10880).  
Currently only the VertexModel is implemented, but we do plan to implement a FaceModel 
as well.

---

## Experiment run

First make sure that you have all the required libraries. All of them can be easily
installed by running the bellow command:
```
pip install -e .
```

Then you can run the training on a collection of simple geometric solids.

```
python experiments/main.py
```
The reconstructions will be saved in the `results` directory.

___

## Object Generation

In order to generate objects use weights obtained during the training.  
**TODO**: Provide a fix.

```
python experiments/generate.py -load_weights <path_to_weights>/epoch_<x>.pt
```