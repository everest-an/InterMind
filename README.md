# InterMind / LatentCore

LatentCore 是一个 **OpenAI 兼容的 Token 压缩网关**，用于在长上下文场景下降低 Token 成本、缓解显存压力，并提供可恢复的隐空间记忆能力。

核心思路：
- 拦截标准 `/v1/chat/completions` 请求
- 自动识别长上下文并压缩为 VQ 离散索引
- 在需要时解压并注入回推理请求

---

## 1. 核心能力

- **OpenAI 兼容代理**：直接接入现有 Agent/SDK，无需改业务逻辑
- **VQ 编码与解码**：`text -> indices` / `indices -> text`
- **Latent 引用机制**：通过 `ref_id` 在后续对话中复用压缩记忆
- **Infini-attention 记忆池**：为长周期任务提供稳定上下文支撑
- **可观测性**：健康检查、codebook 统计、请求级 token 计数中间件

---

## 2. 架构概览

请求流：

1. 客户端发送 `messages` 到 LatentCore
2. 代理层分析上下文长度（Context Analyzer）
3. 长文本进入压缩服务（TextEncoder + VQ + SQLite 存储）
4. 转发到上游 OpenAI 兼容模型服务（如 Ollama/vLLM）
5. 需要召回历史信息时，通过 `ref_id` 解码并注入上下文

关键模块：
- `routes/proxy.py`：OpenAI 兼容入口
- `routes/latent.py`：VQ 编解码与 latent chat 接口
- `services/compression_service.py`：压缩/解压核心服务
- `engine/vq_codebook.py`、`engine/infini_attention.py`：底层向量量化与记忆引擎
- `storage/latent_store.py`：离散索引持久化

---

## 3. 快速开始

### 3.1 环境要求

- Python `>=3.10`
- Windows / Linux / macOS
- 一个 OpenAI 兼容上游服务（默认 `http://localhost:11434/v1`）

### 3.2 安装

```bash
pip install -e .
```

开发依赖：

```bash
pip install -e .[dev]
```

### 3.3 配置

复制环境变量模板：

```bash
copy .env.example .env
```

最关键配置：
- `LATENTCORE_UPSTREAM_BASE_URL`：上游模型 API 地址
- `LATENTCORE_PORT`：网关端口（默认 `8000`）
- `LATENTCORE_COMPRESS_THRESHOLD_TOKENS`：触发压缩阈值
- `LATENTCORE_DB_PATH`：SQLite 文件路径
- `LATENTCORE_DEVICE`：`auto | cpu | cuda | mps`

### 3.4 启动服务

方式一（推荐，已注册 CLI）：

```bash
latentcore
```

方式二：

```bash
python -m latentcore
```

启动后默认地址：`http://0.0.0.0:8000`

---

## 4. API 说明

### 4.1 健康检查

`GET /health`

返回服务状态、上游可达性、设备信息等。

### 4.2 OpenAI 兼容聊天代理

`POST /v1/chat/completions`

支持普通与流式请求，作为现有 OpenAI 客户端的替代 base URL。

### 4.3 文本压缩为离散索引

`POST /v1/latent/encode_vq`

示例：

```bash
curl -X POST "http://localhost:8000/v1/latent/encode_vq" \
	-H "Content-Type: application/json" \
	-d '{
		"text": "这里是一段需要压缩的长文本",
		"session_id": "demo-session"
	}'
```

返回：`ref_id`、`indices`、`compression_ratio` 等。

### 4.4 离散索引解压回文本

`POST /v1/latent/decode_vq`

可通过 `ref_id` 或直接传 `indices` 进行解码。

### 4.5 混合负载聊天（文本 + latent 引用）

`POST /v1/latent/chat_completions`

支持请求体包含 `latent_refs: ["ref_xxx", ...]`，服务会自动恢复上下文并注入到消息中。

### 4.6 Codebook 统计

`GET /v1/latent/codebook/stats`

用于观察向量量化使用情况。

---

## 5. 与 OpenClaw/Agent 系统集成

接入步骤：

1. 启动 LatentCore
2. 将 Agent 的 OpenAI Base URL 指向 `http://localhost:8000/v1`
3. 保持模型与消息协议不变
4. 在长任务中使用 `encode_vq` 生成并复用 `ref_id`

效果：
- 大幅减少重复长文本传输
- 降低上下文窗口占用
- 提升长链路任务稳定性

---

## 6. 测试与质量

运行测试：

```bash
pytest
```

静态检查：

```bash
ruff check src tests
mypy src
```

---

## 7. 项目结构

```text
src/latentcore/
	app.py                 # FastAPI 应用与生命周期
	config.py              # 配置读取
	routes/                # HTTP 路由
	services/              # 业务编排层
	engine/                # 编码、量化、注意力记忆等引擎
	storage/               # SQLite 持久化
	middleware/            # 错误处理、token 计数
tests/                   # 单测与集成测试
docs/                    # 产品与技术设计文档
```

---

## 8. Roadmap（简版）

- 更完整的 OpenAI 响应兼容性（错误码、流式边界）
- 更细粒度压缩策略（按角色/消息段落）
- 多租户与权限隔离
- Latent 资产管理与跨任务复用能力

---

## 9. 许可证

BSD-3-Clause，详见 `LICENSE`。

---

## 10. 创作者

- EverestAn
- EdwinHao
