# RFormer
Official Repository for the NeurIPS 2024 paper **Rough Transformers: Lightweight and Continuous Time Series Modelling through Signature Patching**.

(Note: Description of how to run will come soon. The code will also undergo some refactoring in the near future.)

Please, if you use this code, cite the [published paper in the Proceedings of NeurIPS 2024](https://arxiv.org/abs/2405.20799):

```
@inproceedings{morenorough,
  title={Rough Transformers: Lightweight and Continuous Time Series Modelling through Signature Patching},
  author={Moreno-Pino, Fernando and Arroyo, Alvaro and Waldon, Harrison and Dong, Xiaowen and Cartea, Alvaro},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

## Creating Conda Environment

The repo contains a `rformer.yml` file, a conda environment that allows running the RFormer model. To import and activate it, you can do:

```
conda env create -f rformer.yml
conda activate rformer
```
