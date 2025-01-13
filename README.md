# Cosmological and astrophysical parameter inference using the CAMELS Multifield Dataset

[CMD](https://camels-multifield-dataset.readthedocs.io/en/latest/index.html)

The [CAMELS project](https://camels.readthedocs.io/en/latest/)

### Setup

```shell
pdm install
```

Download dataset files from [url](https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/) and place them under `data/<SIM>/`
 directory.

### Training

Edit param values in `INPUT` section of `fo/scripts/train.py` (lol). Refer to [CMD docs](https://camels-multifield-dataset.readthedocs.io/en/latest/index.html) for details.

Then:

```shell
pdm run train
```
This will start the hyperparam tuning proces using optuna. Checkpointed models will be written to `outputs/models_<SIM>/`.

### Testing

:sparkles: Coming soon :sparkles:

### Troubleshooting

###### `ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory` or `ImportError: libnccl.so.2: cannot open shared object file: No such file or directory`

CUDNN and NCCL libraries could not be found. Since they are installed alongside Torch, you can simply find them in your virtual env and append to `LD_LIBRARY_PATH`:

```shell
libcudnn_dir=$(find $VIRTUAL_ENV -name "libcudnn.so.9" | xargs dirname)
libnccl_dir=$(find $VIRTUAL_ENV -name "libnccl.so.2" | xargs dirname)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$libcudnn_dir:$libnccl_dir
```
