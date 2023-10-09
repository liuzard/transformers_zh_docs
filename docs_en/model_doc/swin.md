<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Swin Transformer

## Overview

The Swin Transformer was proposed in [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.

The abstract from the paper is the following:

*This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone
for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains,
such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text.
To address these differences, we propose a hierarchical Transformer whose representation is computed with \bold{S}hifted
\bold{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping
local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at
various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it
compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense
prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation
(53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and
+2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones.
The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures.*

Tips:
- One can use the [`AutoImageProcessor`] API to prepare images for the model.
- Swin pads the inputs supporting any input height and width (if divisible by `32`).
- Swin can be used as a *backbone*. When `output_hidden_states = True`, it will output both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, sequence_length, num_channels)`.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/swin_transformer_architecture.png"
alt="drawing" width="600"/>

<small> Swin Transformer architecture. Taken from the <a href="https://arxiv.org/abs/2102.03334">original paper</a>.</small>

This model was contributed by [novice03](https://huggingface.co/novice03). The Tensorflow version of this model was contributed by [amyeroberts](https://huggingface.co/amyeroberts). The original code can be found [here](https://github.com/microsoft/Swin-Transformer).


## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Swin Transformer.

<PipelineTag pipeline="image-classification"/>

- [`SwinForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

Besides that:

- [`SwinForMaskedImageModeling`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## SwinConfig

[[autodoc]] SwinConfig


## SwinModel

[[autodoc]] SwinModel
    - forward

## SwinForMaskedImageModeling

[[autodoc]] SwinForMaskedImageModeling
    - forward

## SwinForImageClassification

[[autodoc]] transformers.SwinForImageClassification
    - forward

## TFSwinModel

[[autodoc]] TFSwinModel
    - call

## TFSwinForMaskedImageModeling

[[autodoc]] TFSwinForMaskedImageModeling
    - call

## TFSwinForImageClassification

[[autodoc]] transformers.TFSwinForImageClassification
    - call
