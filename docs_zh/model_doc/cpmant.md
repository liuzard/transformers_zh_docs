<!--版权所有2022年The HuggingFace团队和OpenBMB团队。保留所有权利。

根据Apache许可证，版本2.0（“许可证”），除非符合许可证的要求，否则禁止使用此文件。
你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，不附带任何形式的明示或暗示保证。请参阅许可证了解特定语言的权限和限制。

⚠️ 请注意，该文件是Markdown格式，但包含特定于我们的文档生成器（类似于MDX）的语法，可能无法在你的Markdown查看器中正确呈现。

-->

# CPMAnt

## 概述

CPM-Ant是一个具有100亿参数的开源中文预训练语言模型（PLM）。这也是CPM-Live在线训练过程的第一个里程碑。训练过程具有成本效益和环境友好性。CPM-Ant还通过对CUGE基准的性能调优取得了有希望的结果。除了提供完整模型外，我们还提供了各种压缩版本，以满足不同硬件配置的要求。[查看更多](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live)

提示：

该模型由[OpenBMB](https://huggingface.co/openbmb)贡献。原始代码可以在[此处](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live)找到。

⚙️ 训练与推理
- [CPM-Live教程](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live)。

## CpmAntConfig

[[autodoc]] CpmAntConfig
    - all

## CpmAntTokenizer

[[autodoc]] CpmAntTokenizer
    - all

## CpmAntModel

[[autodoc]] CpmAntModel
    - all
    
## CpmAntForCausalLM

[[autodoc]] CpmAntForCausalLM
    - all