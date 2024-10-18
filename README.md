# G-LeCo

## Requirements:

Ubuntu 22.04.3 LTS

CUDA 12.3 (should be downloaded manually from https://developer.nvidia.com/cuda-12-3-0-download-archive)

To install dependencies, run the following script:

```bash
sudo bash ./scripts/setup.sh
```

## Set up

To have a quick start,you can run:

```bash
./run_var.sh run run
```

Data sets used in the Evaluation part can be downloaded using:

```bash
./script/download_dataset.sh
```

movie_id need to be manually downloaded from the following urls:

```bash
https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv
```

You can download and generate the data sets and store them in a directory `data` under the project root directory.

`linear` and `normal` datasets can be generated by running:

```bash
./scripts/gen_norm.py
```

 