<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepSpeed Integration

[DeepSpeed](https://github.com/microsoft/DeepSpeed) implements everything described in the [ZeRO paper](https://arxiv.org/abs/1910.02054). Currently it provides full support for:

1. Optimizer state partitioning (ZeRO stage 1)
2. Gradient partitioning (ZeRO stage 2)
3. Parameter partitioning (ZeRO stage 3)
4. Custom mixed precision training handling
5. A range of fast CUDA-extension-based optimizers
6. ZeRO-Offload to CPU and NVMe

ZeRO-Offload has its own dedicated paper: [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840). And NVMe-support is described in the paper [ZeRO-Infinity: Breaking the GPU
Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857).

DeepSpeed ZeRO-2 is primarily used only for training, as its features are of no use to inference.

DeepSpeed ZeRO-3 can be used for inference as well, since it allows huge models to be loaded on multiple GPUs, which
won't be possible on a single GPU.

🤗 Transformers integrates [DeepSpeed](https://github.com/microsoft/DeepSpeed) via 2 options:

1. Integration of the core DeepSpeed features via [`Trainer`]. This is an everything-done-for-you type
   of integration - just supply your custom config file or use our template and you have nothing else to do. Most of
   this document is focused on this feature.
2. If you don't use [`Trainer`] and want to use your own Trainer where you integrated DeepSpeed
   yourself, core functionality functions like `from_pretrained` and `from_config` include integration of essential
   parts of DeepSpeed like `zero.Init` for ZeRO stage 3 and higher. To tap into this feature read the docs on
   [non-Trainer DeepSpeed Integration](#nontrainer-deepspeed-integration).

What is integrated:

Training:

1. DeepSpeed ZeRO training supports the full ZeRO stages 1, 2 and 3 with ZeRO-Infinity (CPU and NVME offload).

Inference:

1. DeepSpeed ZeRO Inference supports ZeRO stage 3 with ZeRO-Infinity. It uses the same ZeRO protocol as training, but
   it doesn't use an optimizer and a lr scheduler and only stage 3 is relevant. For more details see:
   [zero-inference](#zero-inference).

There is also DeepSpeed Inference - this is a totally different technology which uses Tensor Parallelism instead of
ZeRO (coming soon).



<a id='deepspeed-trainer-integration'></a>


## Trainer Deepspeed Integration


<a id='deepspeed-installation'></a>

### Installation

Install the library via pypi:

```bash
pip install deepspeed
```

or via `transformers`' `extras`:

```bash
pip install transformers[deepspeed]
```

or find more details on [the DeepSpeed's GitHub page](https://github.com/microsoft/deepspeed#installation) and
[advanced install](https://www.deepspeed.ai/tutorials/advanced-install/).

If you're still struggling with the build, first make sure to read [CUDA Extension Installation Notes](trainer#cuda-extension-installation-notes).

If you don't prebuild the extensions and rely on them to be built at run time and you tried all of the above solutions
to no avail, the next thing to try is to pre-build the modules before installing them.

To make a local build for DeepSpeed:

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

If you intend to use NVMe offload you will also need to include `DS_BUILD_AIO=1` in the instructions above (and also
install *libaio-dev* system-wide).

Edit `TORCH_CUDA_ARCH_LIST` to insert the code for the architectures of the GPU cards you intend to use. Assuming all
your cards are the same you can get the arch via:

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

So if you get `8, 6`, then use `TORCH_CUDA_ARCH_LIST="8.6"`. If you have multiple different cards, you can list all
of them like so `TORCH_CUDA_ARCH_LIST="6.1;8.6"`

If you need to use the same setup on multiple machines, make a binary wheel:

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
python setup.py build_ext -j8 bdist_wheel
```

it will generate something like `dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl` which now you can install
as `pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl` locally or on any other machine.

Again, remember to ensure to adjust `TORCH_CUDA_ARCH_LIST` to the target architectures.

You can find the complete list of NVIDIA GPUs and their corresponding **Compute Capabilities** (same as arch in this
context) [here](https://developer.nvidia.com/cuda-gpus).

You can check the archs pytorch was built with using:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

Here is how to find out the arch for one of the installed GPUs. For example, for GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
print(torch.cuda.get_device_properties(torch.device('cuda')))"
```

If the output is:

```bash
_CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)
```

then you know that this card's arch is `8.6`.

You can also leave `TORCH_CUDA_ARCH_LIST` out completely and then the build program will automatically query the
architecture of the GPUs the build is made on. This may or may not match the GPUs on the target machines, that's why
it's best to specify the desired archs explicitly.

If after trying everything suggested you still encounter build issues, please, proceed with the GitHub Issue of
[Deepspeed](https://github.com/microsoft/DeepSpeed/issues),



<a id='deepspeed-multi-gpu'></a>

### Deployment with multiple GPUs

To deploy the DeepSpeed integration adjust the [`Trainer`] command line arguments to include a new argument `--deepspeed ds_config.json`, where `ds_config.json` is the DeepSpeed configuration file as
   documented [here](https://www.deepspeed.ai/docs/config-json/). The file naming is up to you.
   It's recommended to use DeepSpeed's `add_config_arguments` utility to add the necessary command line arguments to your code.
   For more information please see [DeepSpeed's Argument Parsing](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) doc.

You can use a launcher of your choice here. You can continue using the pytorch launcher:

```bash
torch.distributed.run --nproc_per_node=2 your_program.py <normal cl args> --deepspeed ds_config.json
```
or use the launcher provided by `deepspeed`:

```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json
```

As you can see the arguments aren't the same, but for most needs either of them works. The
full details on how to configure various nodes and GPUs can be found [here](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

When you use the `deepspeed` launcher and you want to use all available gpus you can just omit the `--num_gpus` flag.

Here is an example of running `run_translation.py` under DeepSpeed deploying all available GPUs:

```bash
deepspeed examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

Note that in the DeepSpeed documentation you are likely to see `--deepspeed --deepspeed_config ds_config.json` - i.e.
two DeepSpeed-related arguments, but for the sake of simplicity, and since there are already so many arguments to deal
with, we combined the two into a single argument.

For some practical usage examples, please, see this [post](https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400).



<a id='deepspeed-one-gpu'></a>

### Deployment with one GPU

To deploy DeepSpeed with one GPU adjust the [`Trainer`] command line arguments as follows:

```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

This is almost the same as with multiple-GPUs, but here we tell DeepSpeed explicitly to use just one GPU via
`--num_gpus=1`. By default, DeepSpeed deploys all GPUs it can see on the given node. If you have only 1 GPU to start
with, then you don't need this argument. The following [documentation](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) discusses the launcher options.

Why would you want to use DeepSpeed with just one GPU?

1. It has a ZeRO-offload feature which can delegate some computations and memory to the host's CPU and RAM, and thus
   leave more GPU resources for model's needs - e.g. larger batch size, or enabling a fitting of a very big model which
   normally won't fit.
2. It provides a smart GPU memory management system, that minimizes memory fragmentation, which again allows you to fit
   bigger models and data batches.

While we are going to discuss the configuration in details next, the key to getting a huge improvement on a single GPU
with DeepSpeed is to have at least the following configuration in the configuration file:

```json
{
  "zero_optimization": {
     "stage": 2,
     "offload_optimizer": {
         "device": "cpu",
         "pin_memory": true
     },
     "allgather_partitions": true,
     "allgather_bucket_size": 2e8,
     "reduce_scatter": true,
     "reduce_bucket_size": 2e8,
     "overlap_comm": true,
     "contiguous_gradients": true
  }
}
```

which enables optimizer offload and some other important features. You may experiment with the buffer sizes, you will
find more details in the discussion below.

For a practical usage example of this type of deployment, please, see this [post](https://github.com/huggingface/transformers/issues/8771#issuecomment-759176685).

You may also try the ZeRO-3 with CPU and NVMe offload as explained further in this document.

<!--- TODO: Benchmark whether we can get better performance out of ZeRO-3 vs. ZeRO-2 on a single GPU, and then
recommend ZeRO-3 config as starting one. -->

Notes:

- if you need to run on a specific GPU, which is different from GPU 0, you can't use `CUDA_VISIBLE_DEVICES` to limit
  the visible scope of available GPUs. Instead, you have to use the following syntax:

  ```bash
  deepspeed --include localhost:1 examples/pytorch/translation/run_translation.py ...
  ```

  In this example, we tell DeepSpeed to use GPU 1 (second gpu).



<a id='deepspeed-multi-node'></a>

### Deployment with multiple Nodes

The information in this section isn't not specific to the DeepSpeed integration and is applicable to any multi-node program. But DeepSpeed provides a `deepspeed` launcher that is easier to use than other launchers unless you are in a SLURM environment.

For the duration of this section let's assume that you have 2 nodes with 8 gpus each. And you can reach the first node with `ssh hostname1` and second node with `ssh hostname2`, and both must be able to reach each other via ssh locally without a password. Of course, you will need to rename these host (node) names to the actual host names you are working with.

#### The torch.distributed.run launcher


For example, to use `torch.distributed.run`, you could do:

```bash
python -m torch.distributed.run --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

You have to ssh to each node and run this same command on each one of them! There is no rush, the launcher will wait until both nodes will synchronize.

For more information please see [torchrun](https://pytorch.org/docs/stable/elastic/run.html). Incidentally, this is also the launcher that replaced `torch.distributed.launch` a few pytorch versions back.


#### The deepspeed launcher

To use the `deepspeed` launcher instead, you have to first create a `hostfile` file:

```
hostname1 slots=8
hostname2 slots=8
```
and then you can launch it as:

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
your_program.py <normal cl args> --deepspeed ds_config.json
```

Unlike the `torch.distributed.run` launcher, `deepspeed` will automatically launch this command on both nodes!

For more information please see [Resource Configuration (multi-node)](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).


#### Launching in a SLURM environment

In the SLURM environment the following approach can be used. The following is a slurm script `launch.slurm` which you will need to adapt it to your specific SLURM environment.

```bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

All is left is to schedule it to run:
```bash
sbatch launch.slurm
```

`srun` will take care of launching the program simultaneously on all nodes.


#### Use of Non-shared filesystem

By default DeepSpeed expects that a multi-node environment uses a shared storage. If this is not the case and each node can only see the local filesystem, you need to adjust the config file to include a  [`checkpoint`_section](https://www.deepspeed.ai/docs/config-json/#checkpoint-options) with the following setting:

```json
{
  "checkpoint": {
    "use_node_local_storage": true
  }
}
```

Alternatively, you can also use the [`Trainer`]'s `--save_on_each_node` argument, and the above config will be added automatically for you.


<a id='deepspeed-notebook'></a>

### Deployment in Notebooks

The problem with running notebook cells as a script is that there is no normal `deepspeed` launcher to rely on, so
under certain setups we have to emulate it.

If you're using only 1 GPU, here is how you'd have to adjust your training code in the notebook to use DeepSpeed.

```python
# DeepSpeed requires a distributed environment even when only one process is used.
# This emulates a launcher in the notebook
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# Now proceed as normal, plus pass the deepspeed config file
training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

Note: `...` stands for the normal arguments that you'd pass to the functions.

If you want to use more than 1 GPU, you must use a multi-process environment for DeepSpeed to work. That is, you have
to use the launcher for that purpose and this cannot be accomplished by emulating the distributed environment presented
at the beginning of this section.

If you want to create the config file on the fly in the notebook in the current directory, you could have a dedicated
cell with:

```python no-style
%%bash
cat <<'EOT' > ds_config_zero3.json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT
```

If the training script is in a normal file and not in the notebook cells, you can launch `deepspeed` normally via
shell from a cell. For example, to use `run_translation.py` you would launch it with:

```python no-style
!git clone https://github.com/huggingface/transformers
!cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...
```

or with `%%bash` magic, where you can write a multi-line code for the shell program to run:

```python no-style
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```

In such case you don't need any of the code presented at the beginning of this section.

Note: While `%%bash` magic is neat, but currently it buffers the output so you won't see the logs until the process
completes.




<a id='deepspeed-config'></a>

### Configuration

For the complete guide to the DeepSpeed configuration options that can be used in its configuration file please refer
to the [following documentation](https://www.deepspeed.ai/docs/config-json/).

You can find dozens of DeepSpeed configuration examples that address various practical needs in [the DeepSpeedExamples
repo](https://github.com/microsoft/DeepSpeedExamples):

```bash
git clone https://github.com/microsoft/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
```

Continuing the code from above, let's say you're looking to configure the Lamb optimizer. So you can search through the
example `.json` files with:

```bash
grep -i Lamb $(find . -name '*json')
```

Some more examples are to be found in the [main repo](https://github.com/microsoft/DeepSpeed) as well.

When using DeepSpeed you always need to supply a DeepSpeed configuration file, yet some configuration parameters have
to be configured via the command line. You will find the nuances in the rest of this guide.

To get an idea of what DeepSpeed configuration file looks like, here is one that activates ZeRO stage 2 features,
including optimizer states cpu offload, uses `AdamW` optimizer and `WarmupLR` scheduler and will enable mixed
precision training if `--fp16` is passed:

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}
```

When you execute the program, DeepSpeed will log the configuration it received from the [`Trainer`]
to the console, so you can see exactly what was the final configuration passed to it.



<a id='deepspeed-config-passing'></a>

### Passing Configuration

As discussed in this document normally the DeepSpeed configuration is passed as a path to a json file, but if you're
not using the command line interface to configure the training, and instead instantiate the
[`Trainer`] via [`TrainingArguments`] then for the `deepspeed` argument you can
pass a nested `dict`. This allows you to create the configuration on the fly and doesn't require you to write it to
the file system before passing it to [`TrainingArguments`].

To summarize you can do:

```python
TrainingArguments(..., deepspeed="/path/to/ds_config.json")
```

or:

```python
ds_config_dict = dict(scheduler=scheduler_params, optimizer=optimizer_params)
TrainingArguments(..., deepspeed=ds_config_dict)
```

<a id='deepspeed-config-shared'></a>

### Shared Configuration


<Tip warning={true}>

This section is a must-read

</Tip>

Some configuration values are required by both the [`Trainer`] and DeepSpeed to function correctly,
therefore, to prevent conflicting definitions, which could lead to hard to detect errors, we chose to configure those
via the [`Trainer`] command line arguments.

Additionally, some configuration values are derived automatically based on the model's configuration, so instead of
remembering to manually adjust multiple values, it's the best to let the [`Trainer`] do the majority
of configuration for you.

Therefore, in the rest of this guide you will find a special configuration value: `auto`, which when set will be
automatically replaced with the correct or most efficient value. Please feel free to choose to ignore this
recommendation and set the values explicitly, in which case be very careful that your the
[`Trainer`] arguments and DeepSpeed configurations agree. For example, are you using the same
learning rate, or batch size, or gradient accumulation settings? if these mismatch the training may fail in very
difficult to detect ways. You have been warned.

There are multiple other values that are specific to DeepSpeed-only and those you will have to set manually to suit
your needs.

In your own programs, you can also use the following approach if you'd like to modify the DeepSpeed config as a master
and configure [`TrainingArguments`] based on that. The steps are:

1. Create or load the DeepSpeed configuration to be used as a master configuration
2. Create the [`TrainingArguments`] object based on these values

Do note that some values, such as `scheduler.params.total_num_steps` are calculated by
[`Trainer`] during `train`, but you can of course do the math yourself.

<a id='deepspeed-zero'></a>

### ZeRO

[Zero Redundancy Optimizer (ZeRO)](https://www.deepspeed.ai/tutorials/zero/) is the workhorse of DeepSpeed. It
supports 3 different levels (stages) of optimization. The first one is not quite interesting for scalability purposes,
therefore this document focuses on stages 2 and 3. Stage 3 is further improved by the latest addition of ZeRO-Infinity.
You will find more indepth information in the DeepSpeed documentation.

The `zero_optimization` section of the configuration file is the most important part ([docs](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training)), since that is where you define
which ZeRO stages you want to enable and how to configure them. You will find the explanation for each parameter in the
DeepSpeed docs.

This section has to be configured exclusively via DeepSpeed configuration - the [`Trainer`] provides
no equivalent command line arguments.

Note: currently DeepSpeed doesn't validate parameter names, so if you misspell any, it'll use the default setting for
the parameter that got misspelled. You can watch the DeepSpeed engine start up log messages to see what values it is
going to use.



<a id='deepspeed-zero2-config'></a>

#### ZeRO-2 Config

The following is an example of configuration for ZeRO stage 2:

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    }
}
```

**Performance tuning:**

- enabling `offload_optimizer` should reduce GPU RAM usage (it requires `"stage": 2`)
- `"overlap_comm": true` trades off increased GPU RAM usage to lower all-reduce latency. `overlap_comm` uses 4.5x
  the `allgather_bucket_size` and `reduce_bucket_size` values. So if they are set to 5e8, this requires a 9GB
  footprint (`5e8 x 2Bytes x 2 x 4.5`). Therefore, if you have a GPU with 8GB or less RAM, to avoid getting
  OOM-errors you will need to reduce those parameters to about `2e8`, which would require 3.6GB. You will want to do
  the same on larger capacity GPU as well, if you're starting to hit OOM.
- when reducing these buffers you're trading communication speed to avail more GPU RAM. The smaller the buffer size is,
  the slower the communication gets, and the more GPU RAM will be available to other tasks. So if a bigger batch size is
  important, getting a slightly slower training time could be a good trade.

Additionally, `deepspeed==0.4.4` added a new option `round_robin_gradients` which you can enable with:

```json
{
    "zero_optimization": {
        "round_robin_gradients": true
    }
}
```

This is a stage 2 optimization for CPU offloading that parallelizes gradient copying to CPU memory among ranks by fine-grained gradient partitioning. Performance benefit grows with gradient accumulation steps (more copying between optimizer steps) or GPU count (increased parallelism).


<a id='deepspeed-zero3-config'></a>

#### ZeRO-3 Config

The following is an example of configuration for ZeRO stage 3:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

If you are getting OOMs, because your model or activations don't fit into the GPU memory and you have unutilized CPU
memory offloading the optimizer states and parameters to CPU memory with `"device": "cpu"` may solve this limitation.
If you don't want to offload to CPU memory, use `none` instead of `cpu` for the `device` entry. Offloading to
NVMe is discussed further down.

Pinned memory is enabled with `pin_memory` set to `true`. This feature can improve the throughput at the cost of
making less memory available to other processes. Pinned memory is set aside to the specific process that requested it
and its typically accessed much faster than normal CPU memory.

**Performance tuning:**

- `stage3_max_live_parameters`: `1e9`
- `stage3_max_reuse_distance`: `1e9`

If hitting OOM reduce `stage3_max_live_parameters` and `stage3_max_reuse_distance`. They should have minimal impact
on performance unless you are doing activation checkpointing. `1e9` would consume ~2GB. The memory is shared by
`stage3_max_live_parameters` and `stage3_max_reuse_distance`, so it's not additive, it's just 2GB total.

`stage3_max_live_parameters` is the upper limit on how many full parameters you want to keep on the GPU at any given
time. "reuse distance" is a metric we are using to figure out when will a parameter be used again in the future, and we
use the `stage3_max_reuse_distance` to decide whether to throw away the parameter or to keep it. If a parameter is
going to be used again in near future (less than `stage3_max_reuse_distance`) then we keep it to reduce communication
overhead. This is super helpful when you have activation checkpointing enabled, where we do a forward recompute and
backward passes a single layer granularity and want to keep the parameter in the forward recompute till the backward

The following configuration values depend on the model's hidden size:

- `reduce_bucket_size`: `hidden_size*hidden_size`
- `stage3_prefetch_bucket_size`: `0.9 * hidden_size * hidden_size`
- `stage3_param_persistence_threshold`: `10 * hidden_size`

therefore set these values to `auto` and the [`Trainer`] will automatically assign the recommended
values. But, of course, feel free to set these explicitly as well.

`stage3_gather_16bit_weights_on_model_save` enables model fp16 weights consolidation when model gets saved. With large
models and multiple GPUs this is an expensive operation both in terms of memory and speed. It's currently required if
you plan to resume the training. Watch out for future updates that will remove this limitation and make things more
flexible.

If you're migrating from ZeRO-2 configuration note that `allgather_partitions`, `allgather_bucket_size` and
`reduce_scatter` configuration parameters are not used in ZeRO-3. If you keep these in the config file they will just
be ignored.

- `sub_group_size`: `1e9`

`sub_group_size` controls the granularity in which parameters are updated during optimizer steps. Parameters are
grouped into buckets of `sub_group_size` and each buckets is updated one at a time. When used with NVMe offload in
ZeRO-Infinity, `sub_group_size` therefore controls the granularity in which model states are moved in and out of CPU
memory from NVMe during the optimizer step. This prevents running out of CPU memory for extremely large models.

You can leave `sub_group_size` to its default value of *1e9* when not using NVMe offload. You may want to change its
default value in the following cases:

1. Running into OOM during optimizer step: Reduce `sub_group_size` to reduce memory utilization of temporary buffers
2. Optimizer Step is taking a long time: Increase `sub_group_size` to improve bandwidth utilization as a result of
   the increased data buffers.


#### ZeRO-0 Config

Note that we're listing Stage 0 and 1 last since they are rarely used.

Stage 0 is disabling all types of sharding and just using DeepSpeed as DDP. You can turn it on with:

```json
{
    "zero_optimization": {
        "stage": 0
    }
}
```

This will essentially disable ZeRO without you needing to change anything else.


#### ZeRO-1 Config


Stage 1 is Stage 2 minus gradient sharding. You can always try it to speed things a tiny bit to only shard the optimizer states with:


```json
{
    "zero_optimization": {
        "stage": 1
    }
}
```



<a id='deepspeed-nvme'></a>

### NVMe Support

ZeRO-Infinity allows for training incredibly large models by extending GPU and CPU memory with NVMe memory. Thanks to
smart partitioning and tiling algorithms each GPU needs to send and receive very small amounts of data during
offloading so modern NVMe proved to be fit to allow for an even larger total memory pool available to your training
process. ZeRO-Infinity requires ZeRO-3 enabled.

The following configuration example enables NVMe to offload both optimizer states and the params:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
}
```

You can choose to offload both optimizer states and params to NVMe, or just one of them or none. For example, if you
have copious amounts of CPU memory available, by all means offload to CPU memory only as it'd be faster (hint:
*"device": "cpu"*).

Here is the full documentation for offloading [optimizer states](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) and [parameters](https://www.deepspeed.ai/docs/config-json/#parameter-offloading).

Make sure that your `nvme_path` is actually an NVMe, since it will work with the normal hard drive or SSD, but it'll
be much much slower. The fast scalable training was designed with modern NVMe transfer speeds in mind (as of this
writing one can have ~3.5GB/s read, ~3GB/s write peak speeds).

In order to figure out the optimal `aio` configuration block you must run a benchmark on your target setup, as
[explained here](https://github.com/microsoft/DeepSpeed/issues/998).



<a id='deepspeed-zero2-zero3-performance'></a>

#### ZeRO-2 vs ZeRO-3 Performance

ZeRO-3 is likely to be slower than ZeRO-2 if everything else is configured the same because the former has to gather
model weights in addition to what ZeRO-2 does. If ZeRO-2 meets your needs and you don't need to scale beyond a few GPUs
then you may choose to stick to it. It's important to understand that ZeRO-3 enables a much higher scalability capacity
at a cost of speed.

It's possible to adjust ZeRO-3 configuration to make it perform closer to ZeRO-2:

- set `stage3_param_persistence_threshold` to a very large number - larger than the largest parameter, e.g., `6 * hidden_size * hidden_size`. This will keep the parameters on the GPUs.
- turn off `offload_params` since ZeRO-2 doesn't have that option.

The performance will likely improve significantly with just `offload_params` turned off, even if you don't change
`stage3_param_persistence_threshold`. Of course, these changes will impact the size of the model you can train. So
these help you to trade scalability for speed depending on your needs.



<a id='deepspeed-zero2-example'></a>

#### ZeRO-2 Example

Here is a full ZeRO-2 auto-configuration file `ds_config_zero2.json`:

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

Here is a full ZeRO-2 all-enabled manually set configuration file. It is here mainly for you to see what the typical
values look like, but we highly recommend using the one with multiple `auto` settings in it.

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
```

<a id='deepspeed-zero3-example'></a>

#### ZeRO-3 Example

Here is a full ZeRO-3 auto-configuration file `ds_config_zero3.json`:


```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

Here is a full ZeRO-3 all-enabled manually set configuration file. It is here mainly for you to see what the typical
values look like, but we highly recommend using the one with multiple `auto` settings in it.

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 0.94e6,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
```

#### How to Choose Which ZeRO Stage and Offloads To Use For Best Performance

So now you know there are all these different stages. How to decide which of them to use? This section will attempt to address this question.

In general the following applies:

- Speed-wise (left is faster than right)

Stage 0 (DDP) > Stage 1 > Stage 2 > Stage 2 + offload > Stage 3 > Stage 3 + offloads

- GPU Memory usage-wise (right is more GPU memory efficient than left)

Stage 0 (DDP) < Stage 1 < Stage 2 < Stage 2 + offload < Stage 3 < Stage 3 + offloads

So when you want to get the fastest execution while fitting into minimal number of GPUs, here is the process you could follow. We start with the fastest approach and if running into GPU OOM we then go to the next slower approach, but which will use less GPU memory. And so on and so forth.

First of all set batch size to 1 (you can always use gradient accumulation for any desired effective batch size).

1. Enable `--gradient_checkpointing 1` (HF Trainer) or directly `model.gradient_checkpointing_enable()` - if OOM then
2. Try ZeRO stage 2 first. if OOM then
3. Try ZeRO stage 2 + `offload_optimizer` - if OOM then
4. Switch to ZeRO stage 3 - if OOM then
5. Enable `offload_param` to `cpu` - if OOM then
6. Enable `offload_optimizer` to `cpu` - if OOM then

7. If you still can't fit a batch size of 1 first check various default values and lower them if you can. For example, if you use `generate` and you don't use a wide search beam make it narrower as it'd take a lot of memory.

8. Definitely use mixed half-precision over fp32 - so bf16 on Ampere and higher GPUs and fp16 on older gpu architectures.

9. If you still OOM you could add more hardware or enable ZeRO-Infinity - that is switch offloads `offload_param` and  `offload_optimizer` to `nvme`. You need to make sure it's a very fast nvme. As an anecdote I was able to infer BLOOM-176B on a tiny GPU using ZeRO-Infinity except it was extremely slow. But it worked!

You can, of course, work through these steps in reverse by starting with the most GPU memory efficient config and then going backwards. Or try bi-secting it.

Once you have your batch size 1 not leading to OOM, measure your effective throughput.

Next try to increase the batch size to as large as you can, since the higher the batch size the more efficient the GPUs are as they perform the best when matrices they multiply are huge.

Now the performance optimization game starts. You can turn off some offload features or step down in ZeRO stages and increase/decrease batch size and again measure your effective throughput. Rinse and repeat until satisfied.

Don't spend forever on it, but if you're about to start a 3 months training - do spend a few days on it to find the most effective throughput-wise setup. So that your training cost will be the lowest and you will finish training faster. In the current crazy-paced ML world, if it takes you an extra month to train something you are likely to miss a golden opportunity. Of course, this is only me sharing an observation and in no way I'm trying to rush you. Before beginning to train BLOOM-176B I spent 2 days on this process and was able to increase throughput from 90 to 150 TFLOPs! This effort saved us more than one month of training time.

These notes were written primarily for the training mode, but they should mostly apply for inference as well. For example, during inference Gradient Checkpointing is a no-op since it is only useful during training. Additionally, we found out that if you are doing a multi-GPU inference and not using [DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/), [Accelerate](https://huggingface.co/blog/bloom-inference-pytorch-scripts) should provide a superior performance.


Other quick related performance notes:
- if you are training something from scratch always try to have tensors with shapes that are divisible by 16 (e.g. hidden size). For batch size try divisible by 2 at least. There are [wave and tile quanitization](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/) divisibility that is hardware-specific if you want to squeeze even higher performance from your GPUs.


### Activation Checkpointing or Gradient Checkpointing

Activation checkpointing and gradient checkpointing are two distinct terms that refer to the same methodology. It's very confusing but this is how it is.

Gradient checkpointing allows one to trade speed for GPU memory, which either allows one to overcome a GPU OOM, or increase their batch size, which often leads to a better performance.

HF Transformers models don't know anything about DeepSpeed's activation checkpointing, so if you try to enable that feature in the DeepSpeed config file, nothing will happen.

Therefore you have two ways to take advantage of this very beneficial feature:

1. If you want to use a HF Transformers models you can do `model.gradient_checkpointing_enable()` or use `--gradient_checkpointing` in the HF Trainer, which will automatically enable this for you. `torch.utils.checkpoint` is used there.
2. If you write your own model and you want to use DeepSpeed's activation checkpointing you can use the [API prescribed there](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html). You can also take the HF Transformers modeling code and replace `torch.utils.checkpoint` with the DeepSpeed's API. The latter is more flexible since it allows you to offload the forward activations to the CPU memory instead of recalculating them.


### Optimizer and Scheduler

As long as you don't enable `offload_optimizer` you can mix and match DeepSpeed and HuggingFace schedulers and
optimizers, with the exception of using the combination of HuggingFace scheduler and DeepSpeed optimizer:

| Combos       | HF Scheduler | DS Scheduler |
| HF Optimizer | Yes          | Yes          |
| DS Optimizer | No           | Yes          |

It is possible to use a non-DeepSpeed optimizer when `offload_optimizer` is enabled, as long as it has both CPU and
GPU implementation (except LAMB).




<a id='deepspeed-optimizer'></a>

#### Optimizer


DeepSpeed's main optimizers are Adam, AdamW, OneBitAdam, and Lamb. These have been thoroughly tested with ZeRO and are
thus recommended to be used. It, however, can import other optimizers from `torch`. The full documentation is [here](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters).

If you don't configure the `optimizer` entry in the configuration file, the [`Trainer`] will
automatically set it to `AdamW` and will use the supplied values or the defaults for the following command line
arguments: `--learning_rate`, `--adam_beta1`, `--adam_beta2`, `--adam_epsilon` and `--weight_decay`.

Here is an example of the auto-configured `optimizer` entry for `AdamW`:

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}
```

Note that the command line arguments will set the values in the configuration file. This is so that there is one
definitive source of the values and to avoid hard to find errors when for example, the learning rate is set to
different values in different places. Command line rules. The values that get overridden are:

- `lr` with the value of `--learning_rate`
- `betas` with the value of `--adam_beta1 --adam_beta2`
- `eps` with the value of `--adam_epsilon`
- `weight_decay` with the value of `--weight_decay`

Therefore please remember to tune the shared hyperparameters on the command line.

You can also set the values explicitly:

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": 0.001,
         "betas": [0.8, 0.999],
         "eps": 1e-8,
         "weight_decay": 3e-7
       }
   }
}
```

But then you're on your own synchronizing the [`Trainer`] command line arguments and the DeepSpeed
configuration.

If you want to use another optimizer which is not listed above, you will have to add to the top level configuration.

```json
{
   "zero_allow_untested_optimizer": true
}
```

Similarly to `AdamW`, you can configure other officially supported optimizers. Just remember that those may have different config values. e.g. for Adam you will want `weight_decay` around `0.01`.

Additionally, offload works the best when it's used with Deepspeed's CPU Adam optimizer. If you want to use a different optimizer with offload, since `deepspeed==0.8.3` you need to also add:


```json
{
   "zero_force_ds_cpu_optimizer": false
}
```
to the top level configuration.



<a id='deepspeed-scheduler'></a>

#### Scheduler

DeepSpeed supports `LRRangeTest`, `OneCycle`, `WarmupLR` and `WarmupDecayLR` learning rate schedulers. The full
documentation is [here](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters).

Here is where the schedulers overlap between 🤗 Transformers and DeepSpeed:

- `WarmupLR` via `--lr_scheduler_type constant_with_warmup`
- `WarmupDecayLR` via `--lr_scheduler_type linear`. This is also the default value for `--lr_scheduler_type`,
  therefore, if you don't configure the scheduler this is scheduler that will get configured by default.

If you don't configure the `scheduler` entry in the configuration file, the [`Trainer`] will use
the values of `--lr_scheduler_type`, `--learning_rate` and `--warmup_steps` or `--warmup_ratio` to configure a
🤗 Transformers version of it.

Here is an example of the auto-configured `scheduler` entry for `WarmupLR`:

```json
{
   "scheduler": {
         "type": "WarmupLR",
         "params": {
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

Since *"auto"* is used the [`Trainer`] arguments will set the correct values in the configuration
file. This is so that there is one definitive source of the values and to avoid hard to find errors when, for example,
the learning rate is set to different values in different places. Command line rules. The values that get set are:

- `warmup_min_lr` with the value of `0`.
- `warmup_max_lr` with the value of `--learning_rate`.
- `warmup_num_steps` with the value of `--warmup_steps` if provided. Otherwise will use `--warmup_ratio`
  multiplied by the number of training steps and rounded up.
- `total_num_steps` with either the value of `--max_steps` or if it is not provided, derived automatically at run
  time based on the environment and the size of the dataset and other command line arguments (needed for
  `WarmupDecayLR`).

You can, of course, take over any or all of the configuration values and set those yourself:

```json
{
   "scheduler": {
         "type": "WarmupLR",
         "params": {
             "warmup_min_lr": 0,
             "warmup_max_lr": 0.001,
             "warmup_num_steps": 1000
         }
     }
}
```

But then you're on your own synchronizing the [`Trainer`] command line arguments and the DeepSpeed
configuration.

For example, for `WarmupDecayLR`, you can use the following entry:

```json
{
   "scheduler": {
         "type": "WarmupDecayLR",
         "params": {
             "last_batch_iteration": -1,
             "total_num_steps": "auto",
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

and `total_num_steps`, `warmup_max_lr`, `warmup_num_steps` and `total_num_steps` will be set at loading time.




<a id='deepspeed-fp32'></a>

### fp32 Precision

Deepspeed supports the full fp32 and the fp16 mixed precision.

Because of the much reduced memory needs and faster speed one gets with the fp16 mixed precision, the only time you
will want to not use it is when the model you're using doesn't behave well under this training mode. Typically this
happens when the model wasn't pretrained in the fp16 mixed precision (e.g. often this happens with bf16-pretrained
models). Such models may overflow or underflow leading to `NaN` loss. If this is your case then you will want to use
the full fp32 mode, by explicitly disabling the otherwise default fp16 mixed precision mode with:

```json
{
    "fp16": {
        "enabled": false,
    }
}
```

If you're using the Ampere-architecture based GPU, pytorch version 1.7 and higher will automatically switch to using
the much more efficient tf32 format for some operations, but the results will still be in fp32. For details and
benchmarks, please, see [TensorFloat-32(TF32) on Ampere devices](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices). The document includes
instructions on how to disable this automatic conversion if for some reason you prefer not to use it.

With the 🤗 Trainer you can use `--tf32` to enable it, or disable it with `--tf32 0` or `--no_tf32`. By default the PyTorch default is used.



<a id='deepspeed-amp'></a>

### Automatic Mixed Precision

You can use automatic mixed precision with either a pytorch-like AMP way or the apex-like way:

### fp16

To configure pytorch AMP-like mode with fp16 (float16) set:

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

and the [`Trainer`] will automatically enable or disable it based on the value of
`args.fp16_backend`. The rest of config values are up to you.

This mode gets enabled when `--fp16 --fp16_backend amp` or `--fp16_full_eval` command line args are passed.

You can also enable/disable this mode explicitly:

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

But then you're on your own synchronizing the [`Trainer`] command line arguments and the DeepSpeed
configuration.

Here is the [documentation](https://www.deepspeed.ai/docs/config-json/#fp16-training-options).

### bf16

If bf16 (bfloat16) is desired instead of fp16 then the following configuration section is to be used:

```json
{
    "bf16": {
        "enabled": "auto"
    }
}
```

bf16 has the same dynamic range as fp32 and thus doesn't require loss scaling.

This mode gets enabled when `--bf16` or `--bf16_full_eval` command line args are passed.

You can also enable/disable this mode explicitly:

```json
{
    "bf16": {
        "enabled": true
    }
}
```

<Tip>

As of `deepspeed==0.6.0` the bf16 support is new and experimental.

If you use [gradient accumulation](#gradient-accumulation) with bf16-enabled, you need to be aware that it'll accumulate gradients in bf16, which may not be what you want due to this format's low precision, as it may lead to a lossy accumulation.

A work is being done to fix that and provide an option to use a higher precision `dtype` (fp16 or fp32).

</Tip>


### NCCL Collectives

There is the `dtype` of the training regime and there is a separate `dtype` that is used for communication collectives like various reduction and gathering/scattering operations.

All gather/scatter ops are performed in the same `dtype` the data is in, so if you're using bf16 training regime it gets gathered in bf16 - gathering is a non-lossy operation.

Various reduce operations can be quite lossy, for example when gradients are averaged across multiple-gpus, if the communications are done in fp16 or bf16 the outcome is likely be lossy - since when one ads multiple numbers in low precision the result isn't exact. More so with bf16 as it has a lower precision than fp16. Often fp16 is good enough as the loss is minimal when averaging grads which are typically very small. Therefore, by default for half precision training fp16 is used as the default for reduction operations. But you have full control over this functionality and if you choose you can add a small overhead and ensure that reductions will be using fp32 as the accumulation dtype and only when the result is ready it'll get downcast to the half precision `dtype` you're training in.

In order to override the default you simply add a new configuration entry:

```json
{
    "communication_data_type": "fp32"
}
```
The valid values as of this writing are "fp16", "bfp16", "fp32".

note: stage zero 3 had a bug with regards to bf16 comm dtype that was fixed in `deepspeed==0.8.1`



### apex

To configure apex AMP-like mode set:

```json
"amp": {
    "enabled": "auto",
    "opt_level": "auto"
}
```

and the [`Trainer`] will automatically configure it based on the values of `args.fp16_backend` and
`args.fp16_opt_level`.

This mode gets enabled when `--fp16 --fp16_backend apex --fp16_opt_level 01` command line args are passed.

You can also configure this mode explicitly:

```json
{
    "amp": {
        "enabled": true,
        "opt_level": "O1"
    }
}
```

But then you're on your own synchronizing the [`Trainer`] command line arguments and the DeepSpeed
configuration.

Here is the [documentation](https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options).



<a id='deepspeed-bs'></a>

### Batch Size

To configure batch size, use:

```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

and the [`Trainer`] will automatically set `train_micro_batch_size_per_gpu` to the value of
`args.per_device_train_batch_size` and `train_batch_size` to `args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps`.

You can also set the values explicitly:

```json
{
    "train_batch_size": 12,
    "train_micro_batch_size_per_gpu": 4
}
```

But then you're on your own synchronizing the [`Trainer`] command line arguments and the DeepSpeed
configuration.



<a id='deepspeed-grad-acc'></a>

### Gradient Accumulation

To configure gradient accumulation set:

```json
{
    "gradient_accumulation_steps": "auto"
}
```

and the [`Trainer`] will automatically set it to the value of `args.gradient_accumulation_steps`.

You can also set the value explicitly:

```json
{
    "gradient_accumulation_steps": 3
}
```

But then you're on your own synchronizing the [`Trainer`] command line arguments and the DeepSpeed
configuration.



<a id='deepspeed-grad-clip'></a>

### Gradient Clipping

To configure gradient gradient clipping set:

```json
{
    "gradient_clipping": "auto"
}
```

and the [`Trainer`] will automatically set it to the value of `args.max_grad_norm`.

You can also set the value explicitly:

```json
{
    "gradient_clipping": 1.0
}
```

But then you're on your own synchronizing the [`Trainer`] command line arguments and the DeepSpeed
configuration.



<a id='deepspeed-weight-extraction'></a>

### Getting The Model Weights Out

As long as you continue training and resuming using DeepSpeed you don't need to worry about anything. DeepSpeed stores
fp32 master weights in its custom checkpoint optimizer files, which are `global_step*/*optim_states.pt` (this is glob
pattern), and are saved under the normal checkpoint.

**FP16 Weights:**

When a model is saved under ZeRO-2, you end up having the normal `pytorch_model.bin` file with the model weights, but
they are only the fp16 version of the weights.

Under ZeRO-3, things are much more complicated, since the model weights are partitioned out over multiple GPUs,
therefore `"stage3_gather_16bit_weights_on_model_save": true` is required to get the `Trainer` to save the fp16
version of the weights. If this setting is `False` `pytorch_model.bin` won't be created. This is because by default DeepSpeed's `state_dict` contains a placeholder and not the real weights. If we were to save this `state_dict` it won't be possible to load it back.


```json
{
    "zero_optimization": {
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

**FP32 Weights:**

While the fp16 weights are fine for resuming training, if you finished finetuning your model and want to upload it to
the [models hub](https://huggingface.co/models) or pass it to someone else you most likely will want to get the fp32
weights. This ideally shouldn't be done during training since this is a process that requires a lot of memory, and
therefore best to be performed offline after the training is complete. But if desired and you have plenty of free CPU
memory it can be done in the same training script. The following sections will discuss both approaches.


**Live FP32 Weights Recovery:**

This approach may not work if you model is large and you have little free CPU memory left, at the end of the training.

If you have saved at least one checkpoint, and you want to use the latest one, you can do the following:

```python
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = get_last_checkpoint(trainer.args.output_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

If you're using the `--load_best_model_at_end` class:*~transformers.TrainingArguments* argument (to track the best
checkpoint), then you can finish the training by first saving the final model explicitly and then do the same as above:

```python
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
trainer.deepspeed.save_checkpoint(checkpoint_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

<Tip>

Note, that once `load_state_dict_from_zero_checkpoint` was run, the `model` will no longer be usable in the
DeepSpeed context of the same application. i.e. you will need to re-initialize the deepspeed engine, since
`model.load_state_dict(state_dict)` will remove all the DeepSpeed magic from it. So do this only at the very end
of the training.

</Tip>

Of course, you don't have to use class:*~transformers.Trainer* and you can adjust the examples above to your own
trainer.

If for some reason you want more refinement, you can also extract the fp32 `state_dict` of the weights and apply
these yourself as is shown in the following example:

```python
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)  # already on cpu
model = model.cpu()
model.load_state_dict(state_dict)
```

**Offline FP32 Weights Recovery:**

DeepSpeed creates a special conversion script `zero_to_fp32.py` which it places in the top-level of the checkpoint
folder. Using this script you can extract the weights at any point. The script is standalone and you no longer need to
have the configuration file or a `Trainer` to do the extraction.

Let's say your checkpoint folder looks like this:

```bash
$ ls -l output_dir/checkpoint-1/
-rw-rw-r-- 1 stas stas 1.4K Mar 27 20:42 config.json
drwxrwxr-x 2 stas stas 4.0K Mar 25 19:52 global_step1/
-rw-rw-r-- 1 stas stas   12 Mar 27 13:16 latest
-rw-rw-r-- 1 stas stas 827K Mar 27 20:42 optimizer.pt
-rw-rw-r-- 1 stas stas 231M Mar 27 20:42 pytorch_model.bin
-rw-rw-r-- 1 stas stas  623 Mar 27 20:42 scheduler.pt
-rw-rw-r-- 1 stas stas 1.8K Mar 27 20:42 special_tokens_map.json
-rw-rw-r-- 1 stas stas 774K Mar 27 20:42 spiece.model
-rw-rw-r-- 1 stas stas 1.9K Mar 27 20:42 tokenizer_config.json
-rw-rw-r-- 1 stas stas  339 Mar 27 20:42 trainer_state.json
-rw-rw-r-- 1 stas stas 2.3K Mar 27 20:42 training_args.bin
-rwxrw-r-- 1 stas stas 5.5K Mar 27 13:16 zero_to_fp32.py*
```

In this example there is just one DeepSpeed checkpoint sub-folder *global_step1*. Therefore to reconstruct the fp32
weights just run:

```bash
python zero_to_fp32.py . pytorch_model.bin
```

This is it. `pytorch_model.bin` will now contain the full fp32 model weights consolidated from multiple GPUs.

The script will automatically be able to handle either a ZeRO-2 or ZeRO-3 checkpoint.

`python zero_to_fp32.py -h` will give you usage details.

The script will auto-discover the deepspeed sub-folder using the contents of the file `latest`, which in the current
example will contain `global_step1`.

Note: currently the script requires 2x general RAM of the final fp32 model weights.


### ZeRO-3 and Infinity Nuances

ZeRO-3 is quite different from ZeRO-2 because of its param sharding feature.

ZeRO-Infinity further extends ZeRO-3 to support NVMe memory and multiple other speed and scalability improvements.

While all the efforts were made for things to just work without needing any special changes to your models, in certain
circumstances you may find the following information to be needed.



#### Constructing Massive Models

DeepSpeed/ZeRO-3 can handle models with Trillions of parameters which may not fit onto the existing RAM. In such cases,
but also if you want the initialization to happen much faster, initialize the model using *deepspeed.zero.Init()*
context manager (which is also a function decorator), like so:

```python
from transformers import T5ForConditionalGeneration, T5Config
import deepspeed

with deepspeed.zero.Init():
    config = T5Config.from_pretrained("t5-small")
    model = T5ForConditionalGeneration(config)
```

As you can see this gives you a randomly initialized model.

If you want to use a pretrained model, `model_class.from_pretrained` will activate this feature as long as
`is_deepspeed_zero3_enabled()` returns `True`, which currently is setup by the
[`TrainingArguments`] object if the passed DeepSpeed configuration file contains ZeRO-3 config
section. Thus you must create the [`TrainingArguments`] object **before** calling
`from_pretrained`. Here is an example of a possible sequence:

```python
from transformers import AutoModel, Trainer, TrainingArguments

training_args = TrainingArguments(..., deepspeed=ds_config)
model = AutoModel.from_pretrained("t5-small")
trainer = Trainer(model=model, args=training_args, ...)
```

If you're using the official example scripts and your command line arguments include `--deepspeed ds_config.json`
with ZeRO-3 config enabled, then everything is already done for you, since this is how example scripts are written.

Note: If the fp16 weights of the model can't fit onto the memory of a single GPU this feature must be used.

For full details on this method and other related features please refer to [Constructing Massive Models](https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models).

Also when loading fp16-pretrained models, you will want to tell `from_pretrained` to use
`torch_dtype=torch.float16`. For details, please, see [from_pretrained-torch-dtype](#from_pretrained-torch-dtype).


#### Gathering Parameters

Under ZeRO-3 on multiple GPUs no single GPU has all the parameters unless it's the parameters for the currently
executing layer. So if you need to access all parameters from all layers at once there is a specific method to do it.
Most likely you won't need it, but if you do please refer to [Gathering Parameters](https://deepspeed.readthedocs.io/en/latest/zero3.html#manual-parameter-coordination)

We do however use it internally in several places, one such example is when loading pretrained model weights in
`from_pretrained`. We load one layer at a time and immediately partition it to all participating GPUs, as for very
large models it won't be possible to load it on one GPU and then spread it out to multiple GPUs, due to memory
limitations.

Also under ZeRO-3, if you write your own code and run into a model parameter weight that looks like:

```python
tensor([1.0], device="cuda:0", dtype=torch.float16, requires_grad=True)
```

stress on `tensor([1.])`, or if you get an error where it says the parameter is of size `1`, instead of some much
larger multi-dimensional shape, this means that the parameter is partitioned and what you see is a ZeRO-3 placeholder.



<a id='deepspeed-zero-inference'></a>


### ZeRO Inference

ZeRO Inference uses the same config as ZeRO-3 Training. You just don't need the optimizer and scheduler sections. In
fact you can leave these in the config file if you want to share the same one with the training. They will just be
ignored.

Otherwise you just need to pass the usual [`TrainingArguments`] arguments. For example:

```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --do_eval --deepspeed ds_config.json
```

The only important thing is that you need to use a ZeRO-3 configuration, since ZeRO-2 provides no benefit whatsoever
for the inference as only ZeRO-3 performs sharding of parameters, whereas ZeRO-1 shards gradients and optimizer states.

Here is an example of running `run_translation.py` under DeepSpeed deploying all available GPUs:

```bash
deepspeed examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path t5-small --output_dir output_dir \
--do_eval --max_eval_samples 50 --warmup_steps 50  \
--max_source_length 128 --val_max_target_length 128 \
--overwrite_output_dir --per_device_eval_batch_size 4 \
--predict_with_generate --dataset_config "ro-en" --fp16 \
--source_lang en --target_lang ro --dataset_name wmt16 \
--source_prefix "translate English to Romanian: "
```

Since for inference there is no need for additional large memory used by the optimizer states and the gradients you
should be able to fit much larger batches and/or sequence length onto the same hardware.

Additionally DeepSpeed is currently developing a related product called Deepspeed-Inference which has no relationship
to the ZeRO technology, but instead uses tensor parallelism to scale models that can't fit onto a single GPU. This is a
work in progress and we will provide the integration once that product is complete.


### Memory Requirements

Since Deepspeed ZeRO can offload memory to CPU (and NVMe) the framework provides utils that allow one to tell how much CPU and GPU memory will be needed depending on the number of GPUs being used.

Let's estimate how much memory is needed to finetune "bigscience/T0_3B" on a single GPU:

```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.37GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=1
   15.56GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=0
```

So you can fit it on a single 80GB GPU and no CPU offload, or a tiny 8GB GPU but then need ~60GB of CPU memory. (Remember this is just the memory for params, optimizer states and gradients - you will need a bit more memory for cuda kernels, activations and temps.)

Then it's a tradeoff of cost vs speed. It'll be cheaper to buy/rent a smaller GPU (or less GPUs since you can use multiple GPUs with Deepspeed ZeRO. But then it'll be slower, so even if you don't care about how fast something will be done, the slowdown has a direct impact on the duration of using the GPU and thus bigger cost. So experiment and compare which works the best.

If you have enough GPU memory make sure to disable the CPU/NVMe offload as it'll make everything faster.

For example, let's repeat the same for 2 GPUs:

```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 2 GPUs per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   2.84GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   2.84GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.74GB |  23.58GB | offload_param=none, offload_optimizer=none, zero_init=1
   31.11GB |  23.58GB | offload_param=none, offload_optimizer=none, zero_init=0

```

So here you'd want 2x 32GB GPUs or higher without offloading to CPU.

For full information please see [memory estimators](https://deepspeed.readthedocs.io/en/latest/memory.html).



### Filing Issues

Here is how to file an issue so that we could quickly get to the bottom of the issue and help you to unblock your work.

In your report please always include:

1. the full Deepspeed config file in the report

2. either the command line arguments if you were using the [`Trainer`] or
   [`TrainingArguments`] arguments if you were scripting the Trainer setup yourself. Please do not
   dump the [`TrainingArguments`] as it has dozens of entries that are irrelevant.

3. Output of:

    ```bash
    python -c 'import torch; print(f"torch: {torch.__version__}")'
    python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
    python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'
    ```

4. If possible include a link to a Google Colab notebook that we can reproduce the problem with. You can use this
   [notebook](https://github.com/stas00/porting/blob/master/transformers/deepspeed/DeepSpeed_on_colab_CLI.ipynb) as
   a starting point.

5. Unless it's impossible please always use a standard dataset that we can use and not something custom.

6. If possible try to use one of the existing [examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch) to reproduce the problem with.

Things to consider:

- Deepspeed is often not the cause of the problem.

  Some of the filed issues proved to be Deepspeed-unrelated. That is once Deepspeed was removed from the setup, the
  problem was still there.

  Therefore, if it's not absolutely obvious it's a DeepSpeed-related problem, as in you can see that there is an
  exception and you can see that DeepSpeed modules are involved, first re-test your setup without DeepSpeed in it.
  And only if the problem persists then do mentioned Deepspeed and supply all the required details.

- If it's clear to you that the issue is in the DeepSpeed core and not the integration part, please file the Issue
  directly with [Deepspeed](https://github.com/microsoft/DeepSpeed/). If you aren't sure, please do not worry,
  either Issue tracker will do, we will figure it out once you posted it and redirect you to another Issue tracker if
  need be.



### Troubleshooting

#### the `deepspeed` process gets killed at startup without a traceback

If the `deepspeed` process gets killed at launch time without a traceback, that usually means that the program tried
to allocate more CPU memory than your system has or your process is allowed to allocate and the OS kernel killed that
process. This is because your configuration file most likely has either `offload_optimizer` or `offload_param` or
both configured to offload to `cpu`. If you have NVMe, experiment with offloading to NVMe if you're running under
ZeRO-3. Here is how you can [estimate how much memory is needed for a specific model](https://deepspeed.readthedocs.io/en/latest/memory.html).


#### training and/or eval/predict loss is `NaN`

This often happens when one takes a model pre-trained in bf16 mixed precision mode and tries to use it under fp16 (with or without mixed precision). Most models trained on TPU and often the ones released by Google are in this category (e.g. almost all t5-based models). Here the solution is to either use fp32 or bf16 if your hardware supports it (TPU, Ampere GPUs or newer).

The other problem may have to do with using fp16. When you configure this section:

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

and you see in your log that Deepspeed reports `OVERFLOW!` as follows:

```
0%|                                                                                                                             | 0/189 [00:00<?, ?it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
  1%|▌                                                                                                                    | 1/189 [00:00<01:26,  2.17it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072.0
  1%|█▏
 [...]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 14%|████████████████▌                                                                                                   | 27/189 [00:14<01:13,  2.21it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▏                                                                                                  | 28/189 [00:14<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▊                                                                                                  | 29/189 [00:15<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
[...]
```

that means that the Deepspeed loss scaler can't figure out a scaling co-efficient that overcomes loss overflow.

(the log was massaged to be more readable here.)

In this case you usually need to raise the value of `initial_scale_power`. Setting it to `"initial_scale_power": 32` will typically resolve the problem.



### Notes

- DeepSpeed works with the PyTorch [`Trainer`] but not TF [`TFTrainer`].
- While DeepSpeed has a pip installable PyPI package, it is highly recommended that it gets installed from [source](https://github.com/microsoft/deepspeed#installation) to best match your hardware and also if you need to enable
  certain features, like 1-bit Adam, which aren't available in the pypi distribution.
- You don't have to use the [`Trainer`] to use DeepSpeed with 🤗 Transformers - you can use any model
  with your own trainer, and you will have to adapt the latter according to [the DeepSpeed integration instructions](https://www.deepspeed.ai/getting-started/#writing-deepspeed-models).





## Non-Trainer Deepspeed Integration

The [`~integrations.HfDeepSpeedConfig`] is used to integrate Deepspeed into the 🤗 Transformers core
functionality, when [`Trainer`] is not used. The only thing that it does is handling Deepspeed ZeRO-3 param gathering and automatically splitting the model onto multiple gpus during `from_pretrained` call. Everything else you have to do by yourself.

When using [`Trainer`] everything is automatically taken care of.

When not using [`Trainer`], to efficiently deploy DeepSpeed ZeRO-3, you must instantiate the
[`~integrations.HfDeepSpeedConfig`] object before instantiating the model and keep that object alive.

If you're using Deepspeed ZeRO-1 or ZeRO-2 you don't need to use `HfDeepSpeedConfig` at all.

For example for a pretrained model:

```python
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import deepspeed

ds_config = {...}  # deepspeed config object or path to the file
# must run before instantiating the model to detect zero 3
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
model = AutoModel.from_pretrained("gpt2")
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

or for non-pretrained model:

```python
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel, AutoConfig
import deepspeed

ds_config = {...}  # deepspeed config object or path to the file
# must run before instantiating the model to detect zero 3
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
config = AutoConfig.from_pretrained("gpt2")
model = AutoModel.from_config(config)
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

Please note that if you're not using the [`Trainer`] integration, you're completely on your own. Basically follow the documentation on the [Deepspeed](https://www.deepspeed.ai/) website. Also you have to configure explicitly the config file - you can't use `"auto"` values and you will have to put real values instead.

## HfDeepSpeedConfig

[[autodoc]] integrations.HfDeepSpeedConfig
    - all

### Custom DeepSpeed ZeRO Inference

Here is an example of how one could do DeepSpeed ZeRO Inference without using [`Trainer`] when one can't fit a model onto a single GPU. The solution includes using additional GPUs or/and offloading GPU memory to CPU memory.

The important nuance to understand here is that the way ZeRO is designed you can process different inputs on different GPUs in parallel.

The example has copious notes and is self-documenting.

Make sure to:

1. disable CPU offload if you have enough GPU memory (since it slows things down)
2. enable bf16 if you own an Ampere or a newer GPU to make things faster. If you don't have that hardware you may enable fp16 as long as you don't use any model that was pre-trained in bf16 mixed precision (such as most t5 models). These usually overflow in fp16 and you will see garbage as output.

```python
#!/usr/bin/env python

# This script demonstrates how to use Deepspeed ZeRO in an inference mode when one can't fit a model
# into a single GPU
#
# 1. Use 1 GPU with CPU offload
# 2. Or use multiple GPUs instead
#
# First you need to install deepspeed: pip install deepspeed
#
# Here we use a 3B "bigscience/T0_3B" model which needs about 15GB GPU RAM - so 1 largish or 2
# small GPUs can handle it. or 1 small GPU and a lot of CPU memory.
#
# To use a larger model like "bigscience/T0" which needs about 50GB, unless you have an 80GB GPU -
# you will need 2-4 gpus. And then you can adapt the script to handle more gpus if you want to
# process multiple inputs at once.
#
# The provided deepspeed config also activates CPU memory offloading, so chances are that if you
# have a lot of available CPU memory and you don't mind a slowdown you should be able to load a
# model that doesn't normally fit into a single GPU. If you have enough GPU memory the program will
# run faster if you don't want offload to CPU - so disable that section then.
#
# To deploy on 1 gpu:
#
# deepspeed --num_gpus 1 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# To deploy on 2 gpus:
#
# deepspeed --num_gpus 2 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=2 t0.py


from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = "bigscience/T0_3B"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size

# ds_config notes
#
# - enable bf16 if you use Ampere or higher GPU - this will run in mixed precision and will be
# faster.
#
# - for older GPUs you can enable fp16, but it'll only work for non-bf16 pretrained models - e.g.
# all official t5 models are bf16-pretrained
#
# - set offload_param.device to "none" or completely remove the `offload_param` section if you don't
# - want CPU offload
#
# - if using `offload_param` you can manually finetune stage3_param_persistence_threshold to control
# - which params should remain on gpus - the larger the value the smaller the offload size
#
# For indepth info on Deepspeed config see
# https://huggingface.co/docs/transformers/main/main_classes/deepspeed

# keeping the same format as json for consistency, except it uses lower case for true/false
# fmt: off
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
# fmt: on

# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

# now a model can be loaded.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

# Deepspeed ZeRO can process unrelated inputs on each GPU. So for 2 gpus you process 2 inputs at once.
# If you use more GPUs adjust for more.
# And of course if you have just one input to process you then need to pass the same string to both gpus
# If you use only one GPU, then you will have only rank 0.
rank = torch.distributed.get_rank()
if rank == 0:
    text_in = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
elif rank == 1:
    text_in = "Is this review positive or negative? Review: this is the worst restaurant ever"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, synced_gpus=True)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"rank{rank}:\n   in={text_in}\n  out={text_out}")
```

Let's save it as `t0.py` and run it:
```
$ deepspeed --num_gpus 2 t0.py
rank0:
   in=Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy
  out=Positive
rank1:
   in=Is this review positive or negative? Review: this is the worst restaurant ever
  out=negative
```

This was a very basic example and you will want to adapt it to your needs.

### `generate` nuances

When using multiple GPUs with ZeRO Stage-3, one has to synchronize the GPUs by calling `generate(..., synced_gpus=True)`. If this is not done if one GPU finished generating before other GPUs the whole system will hang as the rest of the GPUs will not be able to received the shard of weights from the GPU that stopped generating.

Starting from `transformers>=4.28`, if `synced_gpus` isn't explicitly specified, it'll be set to `True` automatically if these conditions are detected. But you can still override the value of `synced_gpus` if need to.



## Testing Deepspeed Integration

If you submit a PR that involves DeepSpeed integration please note our CircleCI PR CI setup has no GPUs, so we only run tests requiring gpus on a different CI nightly. Therefore if you get a green CI report in your PR it doesn't mean DeepSpeed tests pass.

To run DeepSpeed tests, please run at least:

```
RUN_SLOW=1 pytest tests/deepspeed/test_deepspeed.py
```

If you changed any of the modeling or pytorch examples code, then run the model zoo tests as well. The following will run all DeepSpeed tests:

```
RUN_SLOW=1 pytest tests/deepspeed
```




## Main DeepSpeed Resources

- [Project's github](https://github.com/microsoft/deepspeed)
- [Usage docs](https://www.deepspeed.ai/getting-started/)
- [API docs](https://deepspeed.readthedocs.io/en/latest/index.html)
- [Blog posts](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

Papers:

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)

Finally, please, remember that, HuggingFace [`Trainer`] only integrates DeepSpeed, therefore if you
have any problems or questions with regards to DeepSpeed usage, please, file an issue with [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/issues).
