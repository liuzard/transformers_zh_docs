<!--版权2020年Marianne J. McQueen.保留所有权利。

根据Apache许可证2.0版（“许可证”）获得许可;除非符合许可证，否则你不得使用此文件。
你可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则以"原样"分发的软件不附带任何明示或暗示的保证或条件。
有关特定语言的特定许可证的详细信息和限制，请参阅许可证。

⚠️注意，此文件以Markdown格式提供，但包含特定于我们的文档生成器（类似于MDX）的语法，可能无法在Markdown查看器中正确呈现。

-->

# BERTology

有一个快速发展的研究领域致力于调查如BERT这样的大规模Transformer的内部工作（一些人称之为“BERTology”）。这个领域有一些很好的例子，包括：

- BERT Rediscovers the Classical NLP Pipeline，作者：Ian Tenney, Dipanjan Das, Ellie Pavlick，链接：https://arxiv.org/abs/1905.05950
- Are Sixteen Heads Really Better than One？作者：Paul Michel, Omer Levy, Graham Neubig，链接：https://arxiv.org/abs/1905.10650
- What Does BERT Look At? An Analysis of BERT's Attention，作者：Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning，链接：https://arxiv.org/abs/1906.04341
- CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure，链接：https://arxiv.org/abs/2210.04633

为了帮助这个新领域的发展，我们在BERT/GPT/GPT-2模型中增加了一些额外的功能，以帮助人们访问内部表示，这些功能主要来自于Paul Michel的杰出工作（https://arxiv.org/abs/1905.10650）：

- 访问BERT/GPT/GPT-2的所有隐藏状态，
- 访问BERT/GPT/GPT-2每个头部的所有注意力权重，
- 检索头部的输出值和梯度，以计算头部的重要性得分和修剪头部，如https://arxiv.org/abs/1905.10650所述。

为了帮助你理解和使用这些功能，我们添加了一个特定的示例脚本：[bertology.py](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology/run_bertology.py)，它提取信息并对在GLUE上预训练的模型进行修剪。