<!--
版权所有 2022 HuggingFace 团队。保留所有权利。

根据 Apache 许可证，版本 2.0 进行许可 (“许可证”)；除非符合许可证的规定，否则不能使用此文件。
您可以在以下位置获取许可以及许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据许可证分发的软件是按“原样”分发的，不带任何明示或默示的保证或条件。
请参阅许可证以了解特定语言下的权限和限制。

️ 注意，此文件是 Markdown 格式的，但包含我们的 doc-builder（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确呈现。

-->

# ESM

## 概述
本页面提供来自 Meta AI 的基础 AI 研究团队的 Transformer 蛋白质语言模型的代码和预训练权重，提供最先进的 ESMFold 和 ESM-2，以及之前发布的 ESM-1b 和 ESM-1v。
Transformer 蛋白质语言模型由 Alexander Rives、Joshua Meier、Tom Sercu、Siddharth Goyal、Zeming Lin、Jason Liu、Demi Guo、Myle Ott、C. Lawrence Zitnick、Jerry Ma 和 Rob Fergus 在论文《[从扩展无监督学习到 2.5 亿种蛋白质序列中的生物结构和功能](https://www.pnas.org/content/118/15/e2016239118)》中介绍。
该论文的第一个版本于 2019 年预印成文。

ESM-2 在各种结构预测任务中表现优于所有经过测试的单序列蛋白质语言模型，并实现了原子分辨率的结构预测。
该模型在 Zeming Lin、Halil Akin、Roshan Rao、Brian Hie、Zhongkai Zhu、Wenting Lu、Allan dos Santos Costa、Maryam Fazel-Zarandi、Tom Sercu、Sal Candido 和 Alexander Rives 的论文《[在演化尺度上蛋白质序列的语言模型可实现准确的结构预测](https://doi.org/10.1101/2022.07.20.500902)》中发布。

该论文还介绍了 ESMFold。它使用 ESM-2 茎和一个能够预测折叠蛋白质结构的头，具有最先进的准确性。与 [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2) 不同，它依赖于来自大规模预训练蛋白质语言模型茎的标记嵌入，并且在推理时不执行多序列比对 (MSA) 步骤，这意味着 ESMFold 检查点是完全“独立”的 - 它们不需要已知蛋白质序列和关联外部查询工具的数据库来进行预测，因此速度更快。

《通过将无监督学习扩展到 2.5 亿种蛋白质序列中形成生物结构和功能》摘要如下：

*在人工智能领域，无监督学习提供的数据规模与模型容量的组合为表示学习和统计生成带来了重大进展。在生命科学中，测序的预期增长承诺提供有关自然序列多样性的前所未有的数据。演化尺度上的蛋白质语言建模是生物学的预测和生成人工智能的逻辑步骤。为此，我们使用无监督学习对跨越演化多样性的 2.5 亿种蛋白质序列上的 860 亿个氨基酸进行了深度上下文语言模型的训练。得到的模型在其表示中包含有关生物性质的信息。这些表示仅通过序列数据进行了学习。学习的表示空间具有多尺度的组织，从氨基酸的生化性质到蛋白质的远源同源性，反映了结构的多级组织。关于二级和三级结构的信息被编码在表示中，并可以通过线性投影进行识别。表示学习产生了具有广泛应用范围的特征，使得能够进行最先进的突变效应和二级结构的监督预测，并改进了用于远程接触的最先进特征。*

《在演化尺度上蛋白质序列的语言模型可实现准确的结构预测》摘要如下：

*最近的研究表明，大型语言模型在规模上发展了随机出现的功能，超越了简单的模式匹配，进行了更高层次的推理，并生成了逼真的图像和文本。虽然已经研究了在较小规模上训练的蛋白质序列语言模型，但我们对它们在规模扩大时对生物学的学习了解甚少。 在本研究中，我们训练了具有 150 亿个参数的模型，这是目前评估过的蛋白质语言模型中最大的模型。我们发现，随着模型的扩展，它们学会了以原子级的分辨率预测蛋白质的三维结构。我们呈现了 ESMFold，它可以直接从蛋白质的序列预测出高准确性的端到端原子级结构。ESMFold 对于那些由语言模型充分理解的困惑度较低的序列，与 AlphaFold2 和 RoseTTAFold 具有相似的准确性。ESMFold 推理速度比 AlphaFold2 快一个数量级，从而可以在实际时间范围内探索宏基因组蛋白质的结构空间。*

提示：

- ESM 模型使用掩码语言建模 (MLM) 目标进行训练。

原始代码可在 [此处](https://github.com/facebookresearch/esm) 找到，由 Meta AI 的基础 AI 研究团队开发。
ESM-1b、ESM-1v 和 ESM-2 的贡献者是 [jasonliu](https://huggingface.co/jasonliu) 和 [Matt](https://huggingface.co/Rocketknight1)。

ESMFold 由 [Matt](https://huggingface.co/Rocketknight1) 和 [Sylvain](https://huggingface.co/sgugger) 贡献，在整个过程中非常感谢 Nikita Smetanin、Roshan Rao 和 Tom Sercu 的帮助！

ESMFold 的 HuggingFace 移植使用了 [openfold](https://github.com/aqlaboratory/openfold) 库的部分内容。
`openfold` 库根据 Apache 许可证 2.0 进行许可。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)

## EsmConfig

[[autodoc]] EsmConfig
    - all

## EsmTokenizer

[[autodoc]] EsmTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## EsmModel

[[autodoc]] EsmModel
    - forward

## EsmForMaskedLM

[[autodoc]] EsmForMaskedLM
    - forward

## EsmForSequenceClassification

[[autodoc]] EsmForSequenceClassification
    - forward

## EsmForTokenClassification

[[autodoc]] EsmForTokenClassification
    - forward

## EsmForProteinFolding

[[autodoc]] EsmForProteinFolding
    - forward

## TFEsmModel

[[autodoc]] TFEsmModel
    - call

## TFEsmForMaskedLM

[[autodoc]] TFEsmForMaskedLM
    - call

## TFEsmForSequenceClassification

[[autodoc]] TFEsmForSequenceClassification
    - call

## TFEsmForTokenClassification

[[autodoc]] TFEsmForTokenClassification
    - call
-->