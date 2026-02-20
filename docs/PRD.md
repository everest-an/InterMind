# 产品需求文档 (PRD)：LatentCore - 基于 LatentMAS 的 OpenClaw 隐空间 Token 压缩网关

## 1. 产品概述与产品愿景

**产品名称：** LatentCore (OpenClaw Token Optimizer)
**产品定位：** 一款完全基于 LatentMAS（隐空间多智能体系统）底层技术，专为 OpenClaw 等本地优先（Local-first）多智能体框架打造的高性能 Token 压缩与显存管理中间件。
**核心愿景：** 彻底消除长周期 AI 任务中的"上下文税"。通过将多智能体交互的媒介从离散的自然语言文本，整体迁移至连续/离散的隐空间张量，为开发者提供极低成本、无遗忘衰减的无限上下文运行环境，并为未来的 AI 记忆资产化（Memory Market）奠定数据标准化底座。

## 2. 市场痛点与问题定义

当前 OpenClaw 等框架在执行复杂任务（如长期代码审计、多轮财报推演）时，面临以下致命瓶颈：

* **心跳轮询导致的 Token 黑洞：** 智能体依赖高频读取本地冗长的 Markdown 记忆、DOM 树和 Shell 日志。随着任务推进，输入序列呈线性甚至指数级增长，导致 API 计费极其昂贵。
* **本地物理显存 (VRAM) 击穿：** 超长上下文的 KV Cache 会迅速耗尽 12GB/24GB 消费级显卡的显存，引发 OOM（Out of Memory）或频繁的 CPU 卸载，导致推理停滞。
* **灾难性遗忘与幻觉：** 传统的纯文本 Summary 会不可逆地丢失早期微小但关键的细节，导致多轮推演后逻辑链断裂。

## 3. 技术底座架构

本产品完全摒弃传统的"大模型二次总结"路线，百分之百依托 LatentMAS 架构，并融合两项前沿注意力与量化机制：

* **纯隐空间流转 (Pure LatentMAS)：** 拦截文本状态，在模型内部（倒数 2-4 层）直接提取并传递高维隐层表征，跳过 Embedding 与 Tokenizer 的重复计算。
* **无限注意力池 (Infini-attention Compressive Memory)：** 将历史隐状态通过线性投影压缩至一个固定大小的全局记忆矩阵中。当前查询（Query）可直接在 O(1) 复杂度下跨时空检索早期特征，彻底解决遗忘问题且锁死 Token 消耗上限。
* **离散化向量量化 (VQ-Latent Codebook)：** 采用 VQ-VAE 机制，将连续的 FP8 张量强制对齐映射为一组离散的整数索引序列。极大降低网络传输带宽与本地存储压力，增强特征的抗噪能力。

---

## 4. 核心功能模块需求

### 4.1 隐空间透明代理层 (API Proxy Interceptor)

* **功能描述：** 作为一个兼容 OpenAI 格式的反向代理服务，无代码侵入地接管 OpenClaw 的 LLM 请求。
* **处理逻辑：**
  1. 拦截 OpenClaw 发送的庞大 `messages` 数组。
  2. 识别并剥离冗长的历史上下文（如系统日志、过往对话）。
  3. 将新指令（文本）与压缩后的隐空间 Codebook 索引打包，送入底层 Latent 引擎进行推理。

### 4.2 LatentCore OpenClaw 原生 Skill 插件

* **功能描述：** 发布在 ClawHub 的标准插件，让 OpenClaw 具备主动管理隐空间记忆的能力。
* **核心动作 (Tools)：**
  * `latent_compress_workspace`: 扫描本地 `.openclaw/memory/` 目录，将超过阈值（如 4000 Tokens）的 Markdown 文件异步推送到云端，替换为极简的 VQ-Latent 离散索引（如 `[LATENT_REF_0xA1B2...]`）。
  * `latent_retrieve_concept`: 当智能体需要回忆特定细节时，发送简短 Query 和索引 ID，直接从 Infini-attention 矩阵中唤醒相关高维特征注入当前推理层。

### 4.3 离散记忆密码本与本地存储库 (VQ-Codebook Manager)

* **功能描述：** 负责管理压缩后的离散化记忆资产。
* **处理逻辑：**
  * 本地不再保存庞大的二进制张量，而是将服务器返回的离散整数数组（体积通常小于原文本的 1/50）存入极其轻量的 SQLite 库或纯文本中。
  * 确保记忆的强一致性，防止在多次心跳传递中发生数值漂移。

### 4.4 隐式记忆流式 API 接口 (RESTful ASGI)

* **接口规范：**
  * `POST /v1/latent/encode_vq`: 接收纯文本，返回基于 Codebook 的离散整数序列。
  * `POST /v1/latent/chat_completions`: 接收混合负载（纯文本 Prompt + VQ 整数序列），调用底层挂载了 Infini-attention 的魔改 vLLM 引擎生成下一帧文本动作。

---

## 5. 用户交互与接入流程

1. **环境配置：** 用户在本地或 VPS 部署 LatentCore 代理服务，或直接获取云端 API Key。
2. **修改网关：** 在 OpenClaw 的配置文件中，将 LLM Base URL 从默认的 `localhost:11434` 修改为 LatentCore 的代理地址 `localhost:8000/v1`。
3. **安装插件：** 在 OpenClaw 终端运行 `claw install latent-context-optimizer`。
4. **无感运行：** OpenClaw 正常启动任务。当其试图写入或读取巨量日志时，代理层和插件自动在后台将其转化为极短的离散索引，终端界面的日志滚动将变得极其精简，同时 Token 消耗仪表的增速呈断崖式下降。

---

## 6. 核心性能指标与验收标准

| 测试领域 | 验收指标与核心要求 | 成功阈值判定 |
| --- | --- | --- |
| **极致压缩率** | 针对 OpenClaw 心跳轮询的等效 Token 消耗缩减 | 较纯文本模式，Token 消耗总量**降低 >= 80%** |
| **记忆保真度** | 在长达 50 轮交互后的"大海捞针"测试 (Needle In A Haystack) | 早期细节召回准确率 **>= 95%**（验证 Infini-attention 效能） |
| **存储极简性** | VQ-Latent 离散化后的本地存储占用 | 1MB 的纯文本记忆，离散化后占用 **< 20KB** |
| **零侵入体验** | OpenClaw 核心框架的兼容性 | 接入代理后，OpenClaw 核心运行逻辑与内置 Skill **0 报错** |

---

## 7. 战略演进：从压缩工具到记忆交易中枢

LatentCore 的终极形态不仅是一个省钱工具，更是未来 AI 记忆交换网络的底层基础设施。通过 VQ-Latent 产生的标准化"离散记忆块"，本质上将无序的 AI 工作状态转化为了可确权、可定价的数字资产。未来，开发者可以将其高质量的 OpenClaw 隐空间记忆打包，直接挂载至去中心化的交易市场，实现 AI 认知价值的自由流通与买卖。
