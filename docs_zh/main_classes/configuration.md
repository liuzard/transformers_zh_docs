<!--版权2020 The HuggingFace团队。版权所有。

根据Apache许可证，第2版（“许可证”）；你不得使用此文件，除非符合许可证的规定。
你可以在以下网站获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件基于“按原样提供”的基础上，不附带任何担保或条件，无论是明示的还是暗示的。请参阅许可证以获取具体语言的权限和限制。

⚠️请注意，本文件采用Markdown格式，但包含特定于我们的文档生成器（类似于MDX）的语法，可能无法在Markdown查看器中正确呈现。

-->

# 配置

基类[`PretrainedConfig`]实现加载/保存配置的公共方法，可以从本地文件或目录加载配置，也可以从库提供的预训练模型配置（从HuggingFace的AWS S3存储库下载）加载。
每个衍生配置类都实现了模型特定的属性。所有配置类中都存在的通用属性包括：`hidden_size`，`num_attention_heads`和`num_hidden_layers`。文本模型还实现了：`vocab_size`。

## 预训练配置

[[autodoc]] PretrainedConfig
    - push_to_hub
    - all