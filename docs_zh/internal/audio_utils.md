<!--版权所有2023 The HuggingFace团队。保留所有权利。

根据Apache License，Version 2.0（“许可证”），您不得使用此文件，除非符合许可证的规定。您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”提供的，不附带任何形式的明示或暗示担保。请查阅许可证以获取许可证下特定语言的权限和限制。

⚠️请注意，该文件采用Markdown格式，但包含我们的文档生成器（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正确呈现。-->

# `FeatureExtractors` 的工具函数

本页列出了所有可由音频 [`FeatureExtractor`] 使用的实用函数，以便使用常见算法（例如*短时傅里叶变换*或*对数梅尔频谱*）从原始音频中计算出特殊特征。

如果您正在研究库中的音频处理器的代码，则大多数都只有在这方面才有用。

## 音频变换

[[autodoc]] audio_utils.hertz_to_mel

[[autodoc]] audio_utils.mel_to_hertz

[[autodoc]] audio_utils.mel_filter_bank

[[autodoc]] audio_utils.optimal_fft_length

[[autodoc]] audio_utils.window_function

[[autodoc]] audio_utils.spectrogram

[[autodoc]] audio_utils.power_to_db

[[autodoc]] audio_utils.amplitude_to_db