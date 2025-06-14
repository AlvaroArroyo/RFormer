# RFormer

Official Repository for the NeurIPS 2024 paper **Rough Transformers: Lightweight and Continuous Time Series Modelling through Signature Patching**.

(Note: The code will undergo some refactoring in the near future.)

Please, if you use this code, cite the [published paper in the Proceedings of NeurIPS 2024](https://arxiv.org/abs/2405.20799):

```
@inproceedings{morenorough,
  title={Rough Transformers: Lightweight and Continuous Time Series Modelling through Signature Patching},
  author={Moreno-Pino, Fernando and Arroyo, Alvaro and Waldon, Harrison and Dong, Xiaowen and Cartea, Alvaro},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

## Requirements

To ensure compatibility, the repository includes a `rformer.yml` file for creating a conda environment with all necessary dependencies.

## Setting Up the Environment

To set up the environment:

```bash
conda env create -f rformer.yml
conda activate rformer
```

## Running the Code

First, clone the repository:
   ```bash
   git clone https://github.com/AlvaroArroyo/RFormer.git
   cd RFormer
   ```

The paper includes experiments on both synthetic (`src/EEG`) and UEA datasets (`src/UEA`). 

To train the model on EEG data without signature preprocessing:

```bash
mkdir src/EEG/data/
python src/synthetic/main_regression_eeg.py --use_signatures
```
To train with signature preprocessing
```bash
mkdir src/EEG/data/
python src/synthetic/main_regression_eeg.py --use_signatures --preprocess
```

To train the model on UEA datasets (https://www.timeseriesclassification.com):

```bash
python src/UEA/main.py --config configs/{config_name}.yaml
```

