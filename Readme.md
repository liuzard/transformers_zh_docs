本项目为🤗HuggingFace transformers 库的中文文档，仅仅针对英文文档进行了翻译工作，版权归HuggingFace 团队所有。

欢迎大家捉虫和贡献。


**1. 开始使用**

- [🤗Transformers简介](docs_zh/index.md)
- [快速开始](docs_zh/quicktour.md)
- [安装](docs_zh/installation.md)

**2. 教程**

- [通过pipline进行推理](docs_zh/pipeline_tutorial.md)

- [使用AutoClass编写可移植代码](docs_zh/autoclass_tutorial.md)

- [数据预处理](docs_zh/preprocessing.md)

- [调优预训练模型](docs_zh/training.md)

- [通过脚本训练](docs_zh/run_scripts.md)

- [通过🤗Accelerate设置分布式训练](docs_zh/accelerate.md)

- [通过🤗PEFT加载和训练adapters](docs_zh/peft.md)

- [分享你的模型](docs_zh/model_sharing.md)


**3. 任务指南**

3.1 自然语言处理

- [文本分类](docs_zh/tasks/sequence_classification.md)

- [Token分类](docs_zh/tasks/token_classification.md)

- [问答](docs_zh/tasks/question_answering.md)

- [语言模型](docs_zh/tasks/language_modeling.md)

- [掩码语言模型](docs_zh/tasks/masked_language_modeling.md)

- [机器翻译](docs_zh/tasks/translation.md)

- [文本摘要](docs_zh/tasks/summarization.md)

- [多项选择](docs_zh/tasks/multiple_choice.md)


3.2 语音处理

- [语音分类](docs_zh/tasks/audio_classification.md)
- [语音识别](docs_zh/tasks/asr.md)

3.3 机器视觉

- [图像分类](docs_zh/transformers_agents.md)
- [图像分割](docs_zh/tasks/semantic_segmentation.md)
- [视频分类](docs_zh/tasks/video_classification.md)
- [目标检测](docs_zh/tasks/object_detection.md)
- [零样本目标检测](docs_zh/tasks/zero_shot_object_detection.md)
- [零样本图像分类](docs_zh/tasks/zero_shot_image_classification.md)
- [深度估计](docs_zh/tasks/monocular_depth_estimation.md)

3.4 多模态

- [图像描述](docs_zh/tasks/image_captioning.md)
- [阅读理解](docs_zh/tasks/document_question_answering.md)
- [图像问答](docs_zh/tasks/visual_question_answering.md)
- [文本转语音](docs_zh/tasks/text-to-speech.md)

3.5 生成

- [自定义生成策略](docs_zh/generation_strategies.md)

3.6 提示

- [通过IDEFICS处理图像任务](docs_zh/tasks/idefics.md)



**4. 开发者指南：**

- [通过🤗Tokenizers实现快速分词](docs_zh/fast_tokenizers.md)
- [多语言模型推理](docs_zh/multilingual.md)
- [使用模型特定的APIs](docs_zh/create_a_model.md)
- [分享自定义的模型](docs_zh/custom_models.md)
- [chat模型模板](docs_zh/chat_templating.md)
- [通过Amazon SageMaker训练模型](docs_zh/serialization.md)
- [导出到ONNX](docs_zh/serialization.md)
- [导出到TFLite](docs_zh/tflite.md)
- [导出到TorchScript](docs_zh/torchscript.md)
- [基准测试](docs_zh/benchmarks.md)
- [笔记样例](docs_zh/notebooks.md)
- [社区资源](docs_zh/community.md)
- [自定义Tools和Prompts](docs_zh/custom_tools.md)
- [问题排查](docs_zh/troubleshooting.md)

**5. 性能和可拓展性：**

[5.1 概述](docs_zh/Overview.md)

5.2 高效训练技巧

- [单个GPU上高效训练的方法和工具](docs_zh/perf_train_gpu_one.md)
- [多个GPU和并行计算](docs_zh/perf_train_gpu_many.md)
- [CPU上高效训练](docs_zh/perf_train_cpu.md)
- [分布式CPU训练](docs_zh/perf_train_cpu_many.md)
- [TPU上训练](docs_zh/perf_train_tpu.md)
- [使用TensorFlow在TPU上训练](docs_zh/perf_train_tpu_tf.md)
- [专用硬件训练](docs_zh/perf_train_special.md)
- [用于训练的自定义硬件](docs_zh/perf_hardware.md)
- [使用Trainer API进行超参数搜索](docs_zh/hpo_train.md)

5.3 优化推理

- [在CPU上进行推理](docs_zh/perf_infer_cpu.md)
- [在单个GPU上进行推理](docs_zh/perf_infer_gpu_one.md)
- [在多个GPU上进行推理](docs_zh/perf_infer_gpu_many.md)
- [在专用硬件上进行推理](docs_zh/perf_infer_special.md)

5.4 其他内容

- [实例化一个大型模型](docs_zh/big_models.md)
- [故障排查](docs_zh/debugging.md)
- [TensorFlow模型的XLA集成](docs_zh/tf_xla.md)
- [使用`torch.compile](docs_zh/.md)`优化推理](docs_zh/perf_torch_compile.md)


**6. 给transformers贡献**

- [如何贡献给🤗Transformers](docs_zh/contributing.md)
- [如何向🤗Transformers添加模型](docs_zh/add_new_model.md)
- [如何将🤗Transformers模型转换为TensorFlow](docs_zh/add_tensorflow_model.md)
- [如何向🤗Transformers添加pipline](docs_zh/add_new_pipeline.md)
- [测试](docs_zh/testing.md)
- [拉取请求的检查](docs_zh/pr_checks.md)

**7. 概念指南**
- [哲学](docs_zh/philosophy.md)
- [术语表](docs_zh/glossary.md)
- [🤗Transformers能做什么](docs_zh/task_summary.md)
- [🤗Transformers如何解决任务](docs_zh/tasks_explained.md)
- [Transformer模型系列](docs_zh/model_summary.md)
- [分词器概述](docs_zh/tokenizer_summary.md)
- [注意力机制](docs_zh/attention.md)
- [填充和截断](docs_zh/pad_truncation.md)
- [BERTology](docs_zh/bertology.md)
- [固定长度模型的困惑度](docs_zh/perplexity.md)
- [用于Web服务器推论的流水线](docs_zh/pipeline_webserver.md)
- [模型训练解剖学](docs_zh/model_memory_anatomy.md)

**7 API**

**7.1 主要的类**
- [Agents and Tools](docs_zh/main_classes/agent.md)
- [Auto Classes](docs_zh/model_doc/auto.md)
- [Callbacks](docs_zh/main_classes/callback.md)
- [Configuration](docs_zh/main_classes/configuration.md)
- [Data Collator](docs_zh/main_classes/data_collator.md)
- [Keras callbacks](docs_zh/main_classes/keras_callbacks.md)
- [Logging](docs_zh/main_classes/logging.md)
- [Models](docs_zh/main_classes/model.md)
- [Text Generation](docs_zh/main_classes/text_generation.md)
- [ONNX](docs_zh/main_classes/onnx.md)
- [Optimization](docs_zh/main_classes/optimizer_schedules.md)
- [Model outputs](docs_zh/main_classes/output.md)
- [Pipelines](docs_zh/main_classes/pipelines.md)
- [Processors](docs_zh/main_classes/processors.md)
- [Quantization](docs_zh/main_classes/quantization.md)
- [Tokenizer](docs_zh/main_classes/tokenizer.md)
- [Trainer](docs_zh/main_classes/trainer.md)
- [DeepSpeed Integration](docs_zh/main_classes/deepspeed.md)
- [Feature Extractor](docs_zh/main_classes/feature_extractor.md)
- [Image Processor](docs_zh/main_classes/image_processor.md)

**7.2 模型**

**7.2.1 文本模型**
- [ALBERT](docs_zh/model_doc/albert.md)
- [BART](docs_zh/model_doc/bart.md)
- [BARThez](docs_zh/model_doc/barthez.md)
- [BARTpho](docs_zh/model_doc/bartpho.md)
- [BERT](docs_zh/model_doc/bert.md)
- [BertGeneration](docs_zh/model_doc/bert-generation.md)
- [BertJapanese](docs_zh/model_doc/bert-japanese.md)
- [Bertweet](docs_zh/model_doc/bertweet.md)
- [BigBird](docs_zh/model_doc/big_bird.md)
- [BigBirdPegasus](docs_zh/model_doc/bigbird_pegasus.md)
- [BioGpt](docs_zh/model_doc/biogpt.md)
- [Blenderbot](docs_zh/model_doc/blenderbot.md)
- [Blenderbot Small](docs_zh/model_doc/blenderbot-small.md)
- [BLOOM](docs_zh/model_doc/bloom.md)
- [BORT](docs_zh/model_doc/bort.md)
- [ByT5](docs_zh/model_doc/byt5.md)
- [CamemBERT](docs_zh/model_doc/camembert.md)
- [CANINE](docs_zh/model_doc/canine.md)
- [CodeGen](docs_zh/model_doc/codegen.md)
- [CodeLlama](docs_zh/model_doc/code_llama.md)
- [ConvBERT](docs_zh/model_doc/convbert.md)
- [CPM](docs_zh/model_doc/cpm.md)
- [CPMANT](docs_zh/model_doc/cpmant.md)
- [CTRL](docs_zh/model_doc/ctrl.md)
- [DeBERTa](docs_zh/model_doc/deberta.md)
- [DeBERTa-v2](docs_zh/model_doc/deberta-v2.md)
- [DialoGPT](docs_zh/model_doc/dialogpt.md)
- [DistilBERT](docs_zh/model_doc/distilbert.md)
- [DPR](docs_zh/model_doc/dpr.md)
- [ELECTRA](docs_zh/model_doc/electra.md)
- [Encoder Decoder Models](docs_zh/model_doc/encoder-decoder.md)
- [ERNIE](docs_zh/model_doc/ernie.md)
- [ErnieM](docs_zh/model_doc/ernie_m.md)
- [ESM](docs_zh/model_doc/esm.md)
- [Falcon](docs_zh/model_doc/falcon.md)
- [FLAN-T5](docs_zh/model_doc/flan-t5.md)
- [FLAN-UL2](docs_zh/model_doc/flan-ul2.md)
- [FlauBERT](docs_zh/model_doc/flaubert.md)
- [FNet](docs_zh/model_doc/fnet.md)
- [FSMT](docs_zh/model_doc/fsmt.md)
- [Funnel Transformer](docs_zh/model_doc/funnel.md)
- [GPT](docs_zh/model_doc/openai-gpt.md)
- [GPT Neo](docs_zh/model_doc/gpt_neo.md)
- [GPT NeoX](docs_zh/model_doc/gpt_neox.md)
- [GPT NeoX Japanese](docs_zh/model_doc/gpt_neox_japanese.md)
- [GPT-J](docs_zh/model_doc/gptj.md)
- [GPT2](docs_zh/model_doc/gpt2.md)
- [GPTBigCode](docs_zh/model_doc/gpt_bigcode.md)
- [GPTSAN Japanese](docs_zh/model_doc/gptsan-japanese.md)
- [GPTSw3](docs_zh/model_doc/gpt-sw3.md)
- [HerBERT](docs_zh/model_doc/herbert.md)
- [I-BERT](docs_zh/model_doc/ibert.md)
- [Jukebox](docs_zh/model_doc/jukebox.md)
- [LED](docs_zh/model_doc/led.md)
- [LLaMA](docs_zh/model_doc/llama.md)
- [Llama2](docs_zh/model_doc/llama2.md)
- [Longformer](docs_zh/model_doc/longformer.md)
- [LongT5](docs_zh/model_doc/longt5.md)
- [LUKE](docs_zh/model_doc/luke.md)
- [M2M100](docs_zh/model_doc/m2m_100.md)
- [MarianMT](docs_zh/model_doc/marian.md)
- [MarkupLM](docs_zh/model_doc/markuplm.md)
- [MBart and MBart-50](docs_zh/model_doc/mbart.md)
- [MEGA](docs_zh/model_doc/mega.md)
- [MegatronBERT](docs_zh/model_doc/megatron-bert.md)
- [MegatronGPT2](docs_zh/model_doc/megatron_gpt2.md)
- [mLUKE](docs_zh/model_doc/mluke.md)
- [MobileBERT](docs_zh/model_doc/mobilebert.md)
- [MPNet](docs_zh/model_doc/mpnet.md)
- [MPT](docs_zh/model_doc/mpt.md)
- [MRA](docs_zh/model_doc/mra.md)
- [MT5](docs_zh/model_doc/mt5.md)
- [MVP](docs_zh/model_doc/mvp.md)
- [NEZHA](docs_zh/model_doc/nezha.md)
- [NLLB](docs_zh/model_doc/nllb.md)
- [NLLB-MoE](docs_zh/model_doc/nllb-moe.md)
- [Nyströmformer](docs_zh/model_doc/nystromformer.md)
- [Open-Llama](docs_zh/model_doc/open-llama.md)
- [OPT](docs_zh/model_doc/opt.md)
- [Pegasus](docs_zh/model_doc/pegasus.md)
- [PEGASUS-X](docs_zh/model_doc/pegasus_x.md)
- [Persimmon](docs_zh/model_doc/persimmon.md)
- [PhoBERT](docs_zh/model_doc/phobert.md)
- [PLBart](docs_zh/model_doc/plbart.md)
- [ProphetNet](docs_zh/model_doc/prophetnet.md)
- [QDQBert](docs_zh/model_doc/qdqbert.md)
- [RAG](docs_zh/model_doc/rag.md)
- [REALM](docs_zh/model_doc/realm.md)
- [Reformer](docs_zh/model_doc/reformer.md)
- [RemBERT](docs_zh/model_doc/rembert.md)
- [RetriBERT](docs_zh/model_doc/retribert.md)
- [RoBERTa](docs_zh/model_doc/roberta.md)
- [RoBERTa-PreLayerNorm](docs_zh/model_doc/roberta-prelayernorm.md)
- [RoCBert](docs_zh/model_doc/roc_bert.md)
- [RoFormer](docs_zh/model_doc/roformer.md)
- [RWKV](docs_zh/model_doc/rwkv.md)
- [Splinter](docs_zh/model_doc/splinter.md)
- [SqueezeBERT](docs_zh/model_doc/squeezebert.md)
- [SwitchTransformers](docs_zh/model_doc/switch_transformers.md)
- [T5](docs_zh/model_doc/t5.md)
- [T5v1.1](docs_zh/model_doc/t5v1.1.md)
- [TAPEX](docs_zh/model_doc/tapex.md)
- [Transformer XL](docs_zh/model_doc/transfo-xl.md)
- [UL2](docs_zh/model_doc/ul2.md)
- [UMT5](docs_zh/model_doc/umt5.md)
- [X-MOD](docs_zh/model_doc/xmod.md)
- [XGLM](docs_zh/model_doc/xglm.md)
- [XLM](docs_zh/model_doc/xlm.md)
- [XLM-ProphetNet](docs_zh/model_doc/xlm-prophetnet.md)
- [XLM-RoBERTa](docs_zh/model_doc/xlm-roberta.md)
- [XLM-RoBERTa-XL](docs_zh/model_doc/xlm-roberta-xl.md)
- [XLM-V](docs_zh/model_doc/xlm-v.md)
- [XLNet](docs_zh/model_doc/xlnet.md)
- [YOSO](docs_zh/model_doc/yoso.md)

**7.2.2 视觉模型**
- [BEiT](docs_zh/model_doc/beit.md)
- [BiT](docs_zh/model_doc/bit.md)
- [Conditional DETR](docs_zh/model_doc/conditional_detr.md)
- [ConvNeXT](docs_zh/model_doc/convnext.md)
- [ConvNeXTV2](docs_zh/model_doc/convnextv2.md)
- [CvT](docs_zh/model_doc/cvt.md)
- [Deformable DETR](docs_zh/model_doc/deformable_detr.md)
- [DeiT](docs_zh/model_doc/deit.md)
- [DETA](docs_zh/model_doc/deta.md)
- [DETR](docs_zh/model_doc/detr.md)
- [DiNAT](docs_zh/model_doc/dinat.md)
- [DINO V2](docs_zh/model_doc/dinov2.md)
- [DiT](docs_zh/model_doc/dit.md)
- [DPT](docs_zh/model_doc/dpt.md)
- [EfficientFormer](docs_zh/model_doc/efficientformer.md)
- [EfficientNet](docs_zh/model_doc/efficientnet.md)
- [FocalNet](docs_zh/model_doc/focalnet.md)
- [GLPN](docs_zh/model_doc/glpn.md)
- [ImageGPT](docs_zh/model_doc/imagegpt.md)
- [LeViT](docs_zh/model_doc/levit.md)
- [Mask2Former](docs_zh/model_doc/mask2former.md)
- [MaskFormer](docs_zh/model_doc/maskformer.md)
- [MobileNetV1](docs_zh/model_doc/mobilenet_v1.md)
- [MobileNetV2](docs_zh/model_doc/mobilenet_v2.md)
- [MobileViT](docs_zh/model_doc/mobilevit.md)
- [MobileViTV2](docs_zh/model_doc/mobilevitv2.md)
- [NAT](docs_zh/model_doc/nat.md)
- [PoolFormer](docs_zh/model_doc/poolformer.md)
- [Pyramid Vision Transformer](docs_zh/model_doc/pvt.md)
- [RegNet](docs_zh/model_doc/regnet.md)
- [ResNet](docs_zh/model_doc/resnet.md)
- [SegFormer](docs_zh/model_doc/segformer.md)
- [SwiftFormer](docs_zh/model_doc/swiftformer.md)
- [Swin Transformer](docs_zh/model_doc/swin.md)
- [Swin Transformer V2](docs_zh/model_doc/swinv2.md)
- [Swin2SR](docs_zh/model_doc/swin2sr.md)
- [Table Transformer](docs_zh/model_doc/table-transformer.md)
- [TimeSformer](docs_zh/model_doc/timesformer.md)
- [UperNet](docs_zh/model_doc/upernet.md)
- [VAN](docs_zh/model_doc/van.md)
- [VideoMAE](docs_zh/model_doc/videomae.md)
- [Vision Transformer](docs_zh/model_doc/vit.md)
- [ViT Hybrid](docs_zh/model_doc/vit_hybrid.md)
- [ViTDet](docs_zh/model_doc/vitdet.md)
- [ViTMAE](docs_zh/model_doc/vit_mae.md)
- [ViTMatte](docs_zh/model_doc/vitmatte.md)
- [ViTMSN](docs_zh/model_doc/vit_msn.md)
- [ViViT](docs_zh/model_doc/vivit.md)
- [YOLOS](docs_zh/model_doc/yolos.md)

**7.2.3 语音模型**
- [Audio Spectrogram Transformer](docs_zh/model_doc/audio-spectrogram-transformer.md)
- [Bark](docs_zh/model_doc/bark.md)
- [CLAP](docs_zh/model_doc/clap.md)
- [EnCodec](docs_zh/model_doc/encodec.md)
- [Hubert](docs_zh/model_doc/hubert.md)
- [MCTCT](docs_zh/model_doc/mctct.md)
- [MMS](docs_zh/model_doc/mms.md)
- [MusicGen](docs_zh/model_doc/musicgen.md)
- [Pop2Piano](docs_zh/model_doc/pop2piano.md)
- [SEW](docs_zh/model_doc/sew.md)
- [SEW-D](docs_zh/model_doc/sew-d.md)
- [Speech2Text](docs_zh/model_doc/speech_to_text.md)
- [Speech2Text2](docs_zh/model_doc/speech_to_text_2.md)
- [SpeechT5](docs_zh/model_doc/speecht5.md)
- [UniSpeech](docs_zh/model_doc/unispeech.md)
- [UniSpeech-SAT](docs_zh/model_doc/unispeech-sat.md)
- [VITS](docs_zh/model_doc/vits.md)
- [Wav2Vec2](docs_zh/model_doc/wav2vec2.md)
- [Wav2Vec2-Conformer](docs_zh/model_doc/wav2vec2-conformer.md)
- [Wav2Vec2Phoneme](docs_zh/model_doc/wav2vec2_phoneme.md)
- [WavLM](docs_zh/model_doc/wavlm.md)
- [Whisper](docs_zh/model_doc/whisper.md)
- [XLS-R](docs_zh/model_doc/xls_r.md)
- [XLSR-Wav2Vec2](docs_zh/model_doc/xlsr_wav2vec2.md)

**7.2.4 多模态模型**
- [ALIGN](docs_zh/model_doc/align.md)
- [AltCLIP](docs_zh/model_doc/altclip.md)
- [BLIP](docs_zh/model_doc/blip.md)
- [BLIP-2](docs_zh/model_doc/blip-2.md)
- [BridgeTower](docs_zh/model_doc/bridgetower.md)
- [BROS](docs_zh/model_doc/bros.md)
- [Chinese-CLIP](docs_zh/model_doc/chinese_clip.md)
- [CLIP](docs_zh/model_doc/clip.md)
- [CLIPSeg](docs_zh/model_doc/clipseg.md)
- [Data2Vec](docs_zh/model_doc/data2vec.md)
- [DePlot](docs_zh/model_doc/deplot.md)
- [Donut](docs_zh/model_doc/donut.md)
- [FLAVA](docs_zh/model_doc/flava.md)
- [GIT](docs_zh/model_doc/git.md)
- [GroupViT](docs_zh/model_doc/groupvit.md)
- [IDEFICS](docs_zh/model_doc/idefics.md)
- [InstructBLIP](docs_zh/model_doc/instructblip.md)
- [LayoutLM](docs_zh/model_doc/layoutlm.md)
- [LayoutLMV2](docs_zh/model_doc/layoutlmv2.md)
- [LayoutLMV3](docs_zh/model_doc/layoutlmv3.md)
- [LayoutXLM](docs_zh/model_doc/layoutxlm.md)
- [LiLT](docs_zh/model_doc/lilt.md)
- [LXMERT](docs_zh/model_doc/lxmert.md)
- [MatCha](docs_zh/model_doc/matcha.md)
- [MGP-STR](docs_zh/model_doc/mgp-str.md)
- [OneFormer](docs_zh/model_doc/oneformer.md)
- [OWL-ViT](docs_zh/model_doc/owlvit.md)
- [Perceiver](docs_zh/model_doc/perceiver.md)
- [Pix2Struct](docs_zh/model_doc/pix2struct.md)
- [Segment Anything](docs_zh/model_doc/sam.md)
- [Speech Encoder Decoder Models](docs_zh/model_doc/speech-encoder-decoder.md)
- [TAPAS](docs_zh/model_doc/tapas.md)
- [TrOCR](docs_zh/model_doc/trocr.md)
- [TVLT](docs_zh/model_doc/tvlt.md)
- [ViLT](docs_zh/model_doc/vilt.md)
- [Vision Encoder Decoder Models](docs_zh/model_doc/vision-encoder-decoder.md)
- [Vision Text Dual Encoder](docs_zh/model_doc/vision-text-dual-encoder.md)
- [VisualBERT](docs_zh/model_doc/visual_bert.md)
- [X-CLIP](docs_zh/model_doc/xclip.md)

**7.2.5 强化学习模型**
- [Decision Transformer](docs_zh/model_doc/decision_transformer.md)
- [Trajectory Transformer](docs_zh/model_doc/trajectory_transformer.md)

**7.2.6 时序模型**
- [Autoformer](docs_zh/model_doc/autoformer.md)
- [Informer](docs_zh/model_doc/informer.md)
- [Time Series Transformer](docs_zh/model_doc/time_series_transformer.md)

**7.2.6 图模型**
- [Graphormer](docs_zh/model_doc/graphormer.md)

**7.3 内部工具**
- [Custom Layers and Utilities](docs_zh/internal/modeling_utils.md)
- [Utilities for pipelines](docs_zh/internal/pipelines_utils.md)
- [Utilities for Tokenizers](docs_zh/internal/tokenization_utils.md)
- [Utilities for Trainer](docs_zh/internal/trainer_utils.md)
- [Utilities for Generation](docs_zh/internal/generation_utils.md)
- [Utilities for Image Processors](docs_zh/internal/image_processing_utils.md)
- [Utilities for Audio processing](docs_zh/internal/audio_utils.md)
- [General Utilities](docs_zh/internal/file_utils.md)
- [Utilities for Time Series](docs_zh/internal/time_series_utils.md)



