# ADMET-AI

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/admet_ai)](https://badge.fury.io/py/admet_ai)
[![PyPI version](https://badge.fury.io/py/admet_ai.svg)](https://badge.fury.io/py/admet_ai)
[![Downloads](https://pepy.tech/badge/admet_ai)](https://pepy.tech/project/admet_ai)
[![license](https://img.shields.io/github/license/swansonk14/admet_ai.svg)](https://github.com/swansonk14/admet_ai/blob/main/LICENSE.txt)

This git repo contains the code for ADMET-AI, an ADMET prediction platform that
uses [Chemprop-RDKit]((https://github.com/chemprop/chemprop)) models trained on ADMET datasets from the Therapeutics
Data Commons ([TDC](https://tdcommons.ai/)). ADMET-AI can be used to make ADMET predictions on new molecules via the
command line, via the Python API, or via a web server. A live web server hosting ADMET-AI is
at [admet.ai.greenstonebio.com](https://admet.ai.greenstonebio.com)

Please see the following paper and [this blog post](https://portal.valencelabs.com/blogs/post/admet-ai-a-machine-learning-admet-platform-for-evaluation-of-large-scale-QPEa0j5OTYYHTaA) for more
details, and please cite us if ADMET-AI is useful in your work. Instructions to reproduce the results in our paper are in [docs/reproduce.md](docs/reproduce.md).

[ADMET-AI: A machine learning ADMET platform for evaluation of large-scale chemical libraries](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btae416/7698030?utm_source=authortollfreelink&utm_campaign=bioinformatics&utm_medium=email&guestAccessKey=f4fca1d2-49ec-4b10-b476-5aea3bf37045)

[Original Repository](https://github.com/swansonk14/admet_ai.git)

## Table of Contents

- [Installation](#installation)
- [Predicting ADMET properties](#predicting-admet-properties)
    * [Command line tool](#command-line-tool)
    * [Python module](#python-module)
    * [Web server](#web-server)

## Installation

ADMET-AI can be installed in a few minutes on any operating system using pip (optionally within a conda environment). If
a GPU is available, it will be used by default, but the code can also run on CPUs only.

Clone the repo and install ADMET-AI locally.

```bash
git clone https://github.com/jjjabcd/admet_ai.git
cd admet_ai
```

Install Environment
```bash
conda create -y -n admet_ai python=3.12
conda activate admet_ai
pip install -r requirements.txt
pip install -e ".[tdc]"
```

Note: If you get the issue `ImportError: libXrender.so.1: cannot open shared object file: No such file or directory`,
run `conda install -c conda-forge xorg-libxrender`.

## Download TDC dataset
```bash
python scripts/prepare_tdc_admet_all.py \
    --save_dir data/tdc_admet_all \
    --skip_datasets herg_central hERG_Karim ToxCast
```

Usage of `--skip_datasets`

The `--skip_datasets` flag is optional. You can choose to use it or not based on your needs:

- Download All: If you omit this flag, the script will download the entire TDC ADMET collection.

- Skip Specifics: Use this flag to exclude certain datasets (e.g., ToxCast) to save download time and disk space, or to avoid redundant data.

### Storage Structure
The data will be organized into subfolders named after each dataset, witch individual CSV files for each label (task).
```
data/tdc_admet_all/
├── Tox21/
│   ├── NR-AhR.csv
│   └── ...
├── Caco2_Wang/
│   └── Caco2_Wang.csv
└── ...
```


## Predicting ADMET properties

ADMET-AI can be used to make ADMET predictions in three ways: (1) as a command line tool, (2) as a Python module, or (3)
as a web server.

### Command line tool

ADMET predictions can be made on the command line with the `admet_predict` command, as illustrated below.

```bash
admet_predict \
    --data_path data.csv \
    --save_path preds.csv \
    --smiles_column smiles
```

This command assumes that there exists a file called `data.csv` with SMILES strings in the column `smiles`. The
predictions will be saved to a file called `preds.csv`.

### Python module

ADMET predictions can be made using the `predict` function in the `admet_ai` Python module, as illustrated below.

```python
from admet_ai import ADMETModel

model = ADMETModel()
preds = model.predict(smiles="O(c1ccc(cc1)CCOC)CC(O)CNC(C)C")
```

If a SMILES string is provided, then `preds` is a dictionary mapping property names to values. If a list of SMILES
strings is provided, then `preds` is a Pandas DataFrame where the index is the SMILES and the columns are the
properties.

### Analysis plots

The DrugBank reference plot and radial plots displayed on the ADMET-AI website can be generated locally using the
`scripts/plot_drugbank_reference.py` and `scripts/plot_radial_summaries.py` scripts, respectively. Both scripts
take as input a CSV file with ADMET-AI predictions along with other parameters.
