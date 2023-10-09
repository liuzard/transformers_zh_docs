<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DINOv2

## Overview

The DINOv2 model was proposed in [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) by
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski.
DINOv2 is an upgrade of [DINO](https://arxiv.org/abs/2104.14294), a self-supervised method applied on [Vision Transformers](vit). This method enables all-purpose visual features, i.e., features that work across image distributions and tasks without finetuning.

The abstract from the paper is the following:

*The recent breakthroughs in natural language processing for model pretraining on large quantities of data have opened the way for similar foundation models in computer vision. These models could greatly simplify the use of images in any system by producing all-purpose visual features, i.e., features that work across image distributions and tasks without finetuning. This work shows that existing pretraining methods, especially self-supervised methods, can produce such features if trained on enough curated data from diverse sources. We revisit existing approaches and combine different techniques to scale our pretraining in terms of data and model size. Most of the technical contributions aim at accelerating and stabilizing the training at scale. In terms of data, we propose an automatic pipeline to build a dedicated, diverse, and curated image dataset instead of uncurated data, as typically done in the self-supervised literature. In terms of models, we train a ViT model (Dosovitskiy et al., 2020) with 1B parameters and distill it into a series of smaller models that surpass the best available all-purpose features, OpenCLIP (Ilharco et al., 2021) on most of the benchmarks at image and pixel levels.*

Tips:

- One can use [`AutoImageProcessor`] class to prepare images for the model.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/facebookresearch/dinov2).


## Dinov2Config

[[autodoc]] Dinov2Config

## Dinov2Model

[[autodoc]] Dinov2Model
    - forward

## Dinov2ForImageClassification

[[autodoc]] Dinov2ForImageClassification
    - forward
