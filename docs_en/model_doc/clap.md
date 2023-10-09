<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CLAP

## Overview

The CLAP model was proposed in [Large Scale Contrastive Language-Audio pretraining with
feature fusion and keyword-to-caption augmentation](https://arxiv.org/pdf/2211.06687.pdf) by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov.

CLAP (Contrastive Language-Audio Pretraining) is a neural network trained on a variety of (audio, text) pairs. It can be instructed in to predict the most relevant text snippet, given an audio, without directly optimizing for the task. The CLAP model uses a SWINTransformer to get audio features from a log-Mel spectrogram input, and a RoBERTa model to get text features. Both the text and audio features are then projected to a latent space with identical dimension. The dot product between the projected audio and text features is then used as a similar score.

The abstract from the paper is the following:

*Contrastive learning has shown remarkable success in the field of multimodal representation learning. In this paper, we propose a pipeline of contrastive language-audio pretraining to develop an audio representation by combining audio data with natural language descriptions. To accomplish this target, we first release LAION-Audio-630K, a large collection of 633,526 audio-text pairs from different data sources. Second, we construct a contrastive language-audio pretraining model by considering different audio encoders and text encoders. We incorporate the feature fusion mechanism and keyword-to-caption augmentation into the model design to further enable the model to process audio inputs of variable lengths and enhance the performance. Third, we perform comprehensive experiments to evaluate our model across three tasks: text-to-audio retrieval, zero-shot audio classification, and supervised audio classification. The results demonstrate that our model achieves superior performance in text-to-audio retrieval task. In audio classification tasks, the model achieves state-of-the-art performance in the zeroshot setting and is able to obtain performance comparable to models' results in the non-zero-shot setting. LAION-Audio-6*

This model was contributed by [Younes Belkada](https://huggingface.co/ybelkada) and [Arthur Zucker](https://huggingface.co/ArtZucker) .
The original code can be found [here](https://github.com/LAION-AI/Clap).


## ClapConfig

[[autodoc]] ClapConfig
    - from_text_audio_configs

## ClapTextConfig

[[autodoc]] ClapTextConfig

## ClapAudioConfig

[[autodoc]] ClapAudioConfig

## ClapFeatureExtractor

[[autodoc]] ClapFeatureExtractor

## ClapProcessor

[[autodoc]] ClapProcessor

## ClapModel

[[autodoc]] ClapModel
    - forward
    - get_text_features
    - get_audio_features

## ClapTextModel

[[autodoc]] ClapTextModel
    - forward

## ClapTextModelWithProjection

[[autodoc]] ClapTextModelWithProjection
    - forward

## ClapAudioModel

[[autodoc]] ClapAudioModel
    - forward

## ClapAudioModelWithProjection

[[autodoc]] ClapAudioModelWithProjection
    - forward

