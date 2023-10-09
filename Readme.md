**开始使用**

-  [🤗 Transformers简介](docs_zh/index.md)
- 快速开始(quicktour)
- 安装(installation)

**教程**

-  通过pipline进行推理 (pipeline_tutorial)

- 使用AutoClass编写可移植代码(autoclass_tutorial)

- 数据预处理(preprocessing)

- 调优预训练模型(training)

- 通过脚本训练(run_scripts)

- 通过🤗 Accelerate设置分布式训练（accelerate）

- 通过 🤗 PEFT加载和训练adapters（peft）

- 分析你的模型（model_sharing）


**任务指南**

1、自然语言处理

- 文本分类（tasks/sequence_classification）

- Token分类（tasks/token_classification）

- 问答（ tasks/question_answering）

- 语言模型（tasks/language_modeling）

- 掩码语言模型（tasks/masked_language_modeling）

- 机器翻译（tasks/translation）

- 文本摘要（tasks/summarization）

- 多项选择（tasks/multiple_choice）


2、语音处理

- 语音分类（tasks/audio_classification）
- 语音识别（tasks/asr）

3、机器视觉

- 图像分类（transformers_agents）
- 图像分割（tasks/semantic_segmentation）
- 视频分类（tasks/video_classification）
- 目标检测（tasks/object_detection）
- 零样本目标检测（tasks/zero_shot_object_detection）
- 零样本图像分类（tasks/zero_shot_image_classification）
- 深度估计（tasks/monocular_depth_estimation）

4、多模态

- 图像描述（Image captioning）
- 阅读理解（tasks/document_question_answering）
- 图像问答（tasks/visual_question_answering）
- 文本转语音（tasks/text-to-speech）

5、生成

- 自定义生成策略（generation_strategies）

6、提示

- 通过IDEFICS处理图像任务（tasks/idefics）



**开发者指南：**

- 通过🤗 Tokenizers实现快速分词
- 多语言模型推理（multilingual）
- 使用模型特定的APIs（create_a_model）
- 分享自定义的模型（custom_models）
- chat模型模板（chat_templating）
- 导出到ONNX（serialization）
- 导出到TFLite（tflite）
- 导出到TorchScript（torchscript）
- 基准测试（benchmarks）
- 笔记样例（notebooks）
- 社区资源（community）
- 自定义Tools和Prompts（custom_tools）
- 问题排查（troubleshooting）

**性能和可拓展性：**

概述（Overview）

1、高效训练技巧

- 单个GPU上高效训练的方法和工具(perf_train_gpu_one)
- 多个GPU和并行计算(perf_train_gpu_many)
- CPU上高效训练(perf_train_cpu)
- 分布式CPU训练(perf_train_cpu_many)
- TPU上训练(perf_train_tpu)
- 使用TensorFlow在TPU上训练(perf_train_tpu_tf)
- 专用硬件训练(perf_train_special)
- 用于训练的自定义硬件(perf_hardware)
- 使用Trainer API进行超参数搜索(hpo_train)

2、优化推理

- 在CPU上进行推理(perf_infer_cpu)
- 在单个GPU上进行推理(perf_infer_gpu_one)
- 在多个GPU上进行推理(perf_infer_gpu_many)
- 在专用硬件上进行推理(perf_infer_special)

3、其他内容

- 实例化一个大型模型(big_models)
- 故障排查(debugging)
- TensorFlow模型的XLA集成(tf_xla)
- 使用`torch.compile()`优化推理(perf_torch_compile)





- 如何贡献给transformers (contributing)
- 如何向🤗 Transformers添加模型 (add_new_model)
- 如何将🤗 Transformers模型转换为TensorFlow (add_tensorflow_model)
- 如何向🤗 Transformers添加流水线 (add_new_pipeline)
- 测试 (testing)
- 拉取请求的检查 (pr_checks)
- 贡献 (Contribute)
- 哲学 (philosophy)
- 术语表 (glossary)
- 🤗 Transformers能做什么 (task_summary)
- 🤗 Transformers如何解决任务 (tasks_explained)
- Transformer模型系列 (model_summary)
- 分词器概述 (tokenizer_summary)
- 注意力机制 (attention)
- 填充和截断 (pad_truncation)
- BERTology (bertology)
- 固定长度模型的困惑度 (perplexity)
- 用于Web服务器推论的流水线 (pipeline_webserver)
- 模型训练解剖学 (model_memory_anatomy)
- 概念指南 (Conceptual guides)

主要的类：

- Agents and Tools (main_classes/agent)
- Auto Classes (model_doc/auto)
- Callbacks (main_classes/callback)
- Configuration (main_classes/configuration)
- Data Collator (main_classes/data_collator)
- Keras callbacks (main_classes/keras_callbacks)
- Logging (main_classes/logging)
- Models (main_classes/model)
- Text Generation (main_classes/text_generation)
- ONNX (main_classes/onnx)
- Optimization (main_classes/optimizer_schedules)
- Model outputs (main_classes/output)
- Pipelines (main_classes/pipelines)
- Processors (main_classes/processors)
- Quantization (main_classes/quantization)
- Tokenizer (main_classes/tokenizer)
- Trainer (main_classes/trainer)
- DeepSpeed Integration (main_classes/deepspeed)
- Feature Extractor (main_classes/feature_extractor)
- Image Processor (main_classes/image_processor)



- ALBERT(model_doc/albert)
- BART(model_doc/bart)
- BARThez(model_doc/barthez)
- BARTpho(model_doc/bartpho)
- BERT(model_doc/bert)
- BertGeneration(model_doc/bert-generation)
- BertJapanese(model_doc/bert-japanese)
- Bertweet(model_doc/bertweet)
- BigBird(model_doc/big_bird)
- BigBirdPegasus(model_doc/bigbird_pegasus)
- BioGpt(model_doc/biogpt)
- Blenderbot(model_doc/blenderbot)
- Blenderbot Small(model_doc/blenderbot-small)
- BLOOM(model_doc/bloom)
- BORT(model_doc/bort)
- ByT5(model_doc/byt5)
- CamemBERT(model_doc/camembert)
- CANINE(model_doc/canine)
- CodeGen(model_doc/codegen)
- CodeLlama(model_doc/code_llama)
- ConvBERT(model_doc/convbert)
- CPM(model_doc/cpm)
- CPMANT(model_doc/cpmant)
- CTRL(model_doc/ctrl)
- DeBERTa(model_doc/deberta)
- DeBERTa-v2(model_doc/deberta-v2)
- DialoGPT(model_doc/dialogpt)
- DistilBERT(model_doc/distilbert)
- DPR(model_doc/dpr)
- ELECTRA(model_doc/electra)
- Encoder Decoder Models(model_doc/encoder-decoder)
- ERNIE(model_doc/ernie)
- ErnieM(model_doc/ernie_m)
- ESM(model_doc/esm)
- Falcon(model_doc/falcon)
- FLAN-T5(model_doc/flan-t5)
- FLAN-UL2(model_doc/flan-ul2)
- FlauBERT(model_doc/flaubert)
- FNet(model_doc/fnet)
- FSMT(model_doc/fsmt)
- Funnel Transformer(model_doc/funnel)
- GPT(model_doc/openai-gpt)
- GPT Neo(model_doc/gpt_neo)
- GPT NeoX(model_doc/gpt_neox)
- GPT NeoX Japanese(model_doc/gpt_neox_japanese)
- GPT-J(model_doc/gptj)
- GPT2(model_doc/gpt2)
- GPTBigCode(model_doc/gpt_bigcode)
- GPTSAN Japanese(model_doc/gptsan-japanese)
- GPTSw3(model_doc/gpt-sw3)
- HerBERT(model_doc/herbert)
- I-BERT(model_doc/ibert)
- Jukebox(model_doc/jukebox)
- LED(model_doc/led)
- LLaMA(model_doc/llama)
- Llama2(model_doc/llama2)
- Longformer(model_doc/longformer)
- LongT5(model_doc/longt5)
- LUKE(model_doc/luke)
- M2M100(model_doc/m2m_100)
- MarianMT(model_doc/marian)
- MarkupLM(model_doc/markuplm)
- MBart and MBart-50(model_doc/mbart)
- MEGA(model_doc/mega)
- MegatronBERT(model_doc/megatron-bert)
- MegatronGPT2(model_doc/megatron_gpt2)
- mLUKE(model_doc/mluke)
- MobileBERT(model_doc/mobilebert)
- MPNet(model_doc/mpnet)
- MPT(model_doc/mpt)
- MRA(model_doc/mra)
- MT5(model_doc/mt5)
- MVP(model_doc/mvp)
- NEZHA(model_doc/nezha)
- NLLB(model_doc/nllb)
- NLLB-MoE(model_doc/nllb-moe)
- Nyströmformer(model_doc/nystromformer)
- Open-Llama(model_doc/open-llama)
- OPT(model_doc/opt)
- Pegasus(model_doc/pegasus)
- PEGASUS-X(model_doc/pegasus_x)
- Persimmon(model_doc/persimmon)
- PhoBERT(model_doc/phobert)
- PLBart(model_doc/plbart)
- ProphetNet(model_doc/prophetnet)
- QDQBert(model_doc/qdqbert)
- RAG(model_doc/rag)
- REALM(model_doc/realm)
- Reformer(model_doc/reformer)
- RemBERT(model_doc/rembert)
- RetriBERT(model_doc/retribert)
- RoBERTa(model_doc/roberta)
- RoBERTa-PreLayerNorm(model_doc/roberta-prelayernorm)
- RoCBert(model_doc/roc_bert)
- RoFormer(model_doc/roformer)
- RWKV(model_doc/rwkv)
- Splinter(model_doc/splinter)
- SqueezeBERT(model_doc/squeezebert)
- SwitchTransformers(model_doc/switch_transformers)
- T5(model_doc/t5)
- T5v1.1(model_doc/t5v1.1)
- TAPEX(model_doc/tapex)
- Transformer XL(model_doc/transfo-xl)
- UL2(model_doc/ul2)
- UMT5(model_doc/umt5)
- X-MOD(model_doc/xmod)
- XGLM(model_doc/xglm)
- XLM(model_doc/xlm)
- XLM-ProphetNet(model_doc/xlm-prophetnet)
- XLM-RoBERTa(model_doc/xlm-roberta)
- XLM-RoBERTa-XL(model_doc/xlm-roberta-xl)
- XLM-V(model_doc/xlm-v)
- XLNet(model_doc/xlnet)
- YOSO(model_doc/yoso)
- BEiT(model_doc/beit)
- BiT(model_doc/bit)
- Conditional DETR(model_doc/conditional_detr)
- ConvNeXT(model_doc/convnext)
- ConvNeXTV2(model_doc/convnextv2)
- CvT(model_doc/cvt)
- Deformable DETR(model_doc/deformable_detr)
- DeiT(model_doc/deit)
- DETA(model_doc/deta)
- DETR(model_doc/detr)
- DiNAT(model_doc/dinat)
- DINO V2(model_doc/dinov2)
- DiT(model_doc/dit)
- DPT(model_doc/dpt)
- EfficientFormer(model_doc/efficientformer)
- EfficientNet(model_doc/efficientnet)
- FocalNet(model_doc/focalnet)
- GLPN(model_doc/glpn)
- ImageGPT(model_doc/imagegpt)
- LeViT(model_doc/levit)
- Mask2Former(model_doc/mask2former)
- MaskFormer(model_doc/maskformer)
- MobileNetV1(model_doc/mobilenet_v1)
- MobileNetV2(model_doc/mobilenet_v2)
- MobileViT(model_doc/mobilevit)
- MobileViTV2(model_doc/mobilevitv2)
- NAT(model_doc/nat)
- PoolFormer(model_doc/poolformer)
- Pyramid Vision Transformer (PVT)(model_doc/pvt)
- RegNet(model_doc/regnet)
- ResNet(model_doc/resnet)
- SegFormer(model_doc/segformer)
- SwiftFormer(model_doc/swiftformer)
- Swin Transformer(model_doc/swin)
- Swin Transformer V2(model_doc/swinv2)
- Swin2SR(model_doc/swin2sr)
- Table Transformer(model_doc/table-transformer)
- TimeSformer(model_doc/timesformer)
- UperNet(model_doc/upernet)
- VAN(model_doc/van)
- VideoMAE(model_doc/videomae)
- Vision Transformer (ViT)(model_doc/vit)
- ViT Hybrid(model_doc/vit_hybrid)
- ViTDet(model_doc/vitdet)
- ViTMAE(model_doc/vit_mae)
- ViTMatte(model_doc/vitmatte)
- ViTMSN(model_doc/vit_msn)
- ViViT(model_doc/vivit)
- YOLOS(model_doc/yolos)
- Audio Spectrogram Transformer(model_doc/audio-spectrogram-transformer)
- Bark(model_doc/bark)
- CLAP(model_doc/clap)
- EnCodec(model_doc/encodec)
- Hubert(model_doc/hubert)
- MCTCT(model_doc/mctct)
- MMS(model_doc/mms)
- MusicGen(model_doc/musicgen)
- Pop2Piano(model_doc/pop2piano)
- SEW(model_doc/sew)
- SEW-D(model_doc/sew-d)
- Speech2Text(model_doc/speech_to_text)
- Speech2Text2(model_doc/speech_to_text_2)
- SpeechT5(model_doc/speecht5)
- UniSpeech(model_doc/unispeech)
- UniSpeech-SAT(model_doc/unispeech-sat)
- VITS(model_doc/vits)
- Wav2Vec2(model_doc/wav2vec2)
- Wav2Vec2-Conformer(model_doc/wav2vec2-conformer)
- Wav2Vec2Phoneme(model_doc/wav2vec2_phoneme)
- WavLM(model_doc/wavlm)
- Whisper(model_doc/whisper)
- XLS-R(model_doc/xls_r)
- XLSR-Wav2Vec2(model_doc/xlsr_wav2vec2)
- ALIGN(model_doc/align)
- AltCLIP(model_doc/altclip)
- BLIP(model_doc/blip)
- BLIP-2(model_doc/blip-2)
- BridgeTower(model_doc/bridgetower)
- BROS(model_doc/bros)
- Chinese-CLIP(model_doc/chinese_clip)
- CLIP(model_doc/clip)
- CLIPSeg(model_doc/clipseg)
- Data2Vec(model_doc/data2vec)
- DePlot(model_doc/deplot)
- Donut(model_doc/donut)
- FLAVA(model_doc/flava)
- GIT(model_doc/git)
- GroupViT(model_doc/groupvit)
- IDEFICS(model_doc/idefics)
- InstructBLIP(model_doc/instructblip)
- LayoutLM(model_doc/layoutlm)
- LayoutLMV2(model_doc/layoutlmv2)
- LayoutLMV3(model_doc/layoutlmv3)
- LayoutXLM(model_doc/layoutxlm)
- LiLT(model_doc/lilt)
- LXMERT(model_doc/lxmert)
- MatCha(model_doc/matcha)
- MGP-STR(model_doc/mgp-str)
- OneFormer(model_doc/oneformer)
- OWL-ViT(model_doc/owlvit)
- Perceiver(model_doc/perceiver)
- Pix2Struct(model_doc/pix2struct)
- Segment Anything(model_doc/sam)
- Speech Encoder Decoder Models(model_doc/speech-encoder-decoder)
- TAPAS(model_doc/tapas)
- TrOCR(model_doc/trocr)
- TVLT(model_doc/tvlt)
- ViLT(model_doc/vilt)
- Vision Encoder Decoder Models(model_doc/vision-encoder-decoder)
- Vision Text Dual Encoder(model_doc/vision-text-dual-encoder)
- VisualBERT(model_doc/visual_bert)
- X-CLIP(model_doc/xclip)
- Decision Transformer(model_doc/decision_transformer)
- Trajectory Transformer(model_doc/trajectory_transformer)
- Autoformer(model_doc/autoformer)
- Informer(model_doc/informer)
- Time Series Transformer(model_doc/time_series_transformer)
- Graphormer(model_doc/graphormer)
- Custom Layers and Utilities(internal/modeling_utils)
- Utilities for pipelines(internal/pipelines_utils)
- Utilities for Tokenizers(internal/tokenization_utils)
- Utilities for Trainer(internal/trainer_utils)
- Utilities for Generation(internal/generation_utils)
- Utilities for Image Processors(internal/image_processing_utils)
- Utilities for Audio processing(internal/audio_utils)
- General Utilities(internal/file_utils)
- Utilities for Time Series(internal/time_series_utils)



