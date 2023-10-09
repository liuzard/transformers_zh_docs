<!--版权所有 © 2022 The HuggingFace团队。

根据Apache License，Version 2.0许可证（以下简称“许可证”）进行许可。你不得使用此文件，除非符合许可证的使用条件。
你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或以书面形式达成协议，本软件按“原样”分发，不对任何种类的保证或条件提供保证。请注意，此文件是Markdown格式，但包含特定语法以供我们的文档构建器（类似于MDX）使用，可能在你的Markdown查看器中无法正确呈现。

-->

# 在CPU上高效训练

本指南专注于在CPU上高效训练大型模型。

## IPEX的混合精度

IPEX针对具有AVX-512或更高版本的CPU进行了优化，并且对于仅具有AVX2的CPU也具有功能性工作。因此，预计它能够在具有AVX-512或更高版本的英特尔CPU代数中带来性能优化，而仅具有AVX2的CPU（例如，AMD CPU或旧英特尔CPU）可能会在IPEX下取得更好的性能，但不能保证。IPEX可为使用Float32和BFloat16进行CPU训练提供性能优化。接下来的几节将重点介绍如何使用BFloat16。

低精度数据类型BFloat16已在第三代Xeon®可扩展处理器（也称为Cooper Lake）上获得了原生支持，并将在下一代Intel® Xeon®可扩展处理器（搭载Intel® Advanced Matrix Extensions（Intel® AMX）指令集）上获得支持，从而进一步提高性能。自PyTorch-1.10起，启用了CPU后端的自动混合精度。同时，在Intel® Extension for PyTorch中大规模启用了对CPU和运算符进行自动混合精度和BFloat16优化的支持，并将其部分上游至PyTorch主分支。用户可以通过使用IPEX自动混合精度来获得更好的性能和用户体验。

更多关于[自动混合精度](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/amp.html)的详细信息请查看。

### IPEX安装：

IPEX的发布遵循PyTorch的发布。使用pip进行安装：

| PyTorch版本   | IPEX版本   |
| :---------------: | :----------:   |
| 1.13              |  1.13.0+cpu    |
| 1.12              |  1.12.300+cpu  |
| 1.11              |  1.11.200+cpu  |
| 1.10              |  1.10.100+cpu  |

```
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

更多关于[IPEX安装](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html)的方法请查看。

### 在训练器中的使用
要在训练器中启用IPEX的自动混合精度，请在训练命令参数中添加`use_ipex`、`bf16`和`no_cuda`。

以使用用例[Transformers question-answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)为例：

- 在CPU上使用IPEX与BF16的自动混合精度进行训练：
<pre> python run_qa.py \
--model_name_or_path bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/debug_squad/ \
<b>--use_ipex \</b>
<b>--bf16 --no_cuda</b></pre> 

### 实践示例

博客：[使用Intel Sapphire Rapids加速PyTorch Transformers](https://huggingface.co/blog/intel-sapphire-rapids)