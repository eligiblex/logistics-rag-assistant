# 物流 RAG 与异常处理 Agent

[![CI](https://github.com/eligiblex/logistics-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/eligiblex/logistics-rag-assistant/actions/workflows/ci.yml)

基于 **LangChain + Ollama + Chroma + FastAPI** 的本地物流 AI 应用。除了带来源的 RAG 问答，还包含一个可调试的异常处理 Agent：它会规划白名单工具、查询演示运单、评估异常，并在创建工单等动作前暂停等待人工审批。项目定位是可复现的学习原型，不是生产级物流平台，不能查询真实订单轨迹或实时运价。

## 功能与工程设计

- 导入 UTF-8 TXT、Markdown 和可提取文本的 PDF；递归分块，保留相对文件名、块编号及 PDF 页码。
- 使用 Ollama Embedding 和 Chroma 持久化向量索引；内容 ID 去重，重复刷新不重复插入，更新时清理旧块。
- **一次检索同时供生成和来源展示使用**，避免回答与参考片段来自两次不同检索。
- 普通 JSON 与 SSE 两种查询方式；SSE 依次传输来源、文本增量、完成状态。
- 浏览器正确缓冲跨包 UTF-8 和半条 SSE 消息；使用 textContent 展示模型输出，避免把输出当 HTML 执行。
- Pydantic 输入校验、配置覆盖、错误处理、刷新互斥与状态查询；18 项 Python 测试及 Node 流式解析测试。
- 显式状态机 Agent：工具计划、执行结果、审批点和最终答复都有 Trace；SQLite 保存运行状态，支持服务重启后恢复审批。
- 工单、投诉、理赔类动作必须人工批准；动作只生成带 `DEMO-` 前缀的本地演示工单，不会调用真实外部系统。

## 数据流

```text
TXT / Markdown / PDF → 加载与分块 → Ollama Embedding → Chroma 持久化
                                                          ↑
问题 → 参数校验 → Top-k 检索 → 带编号的上下文 → Qwen2.5 → 回答
                              └──────────────→ 同一批参考片段
```

默认单轮问答。页面保留显示记录，但历史消息和 session_id 不参与下一次检索；不要将其描述为已实现对话记忆。

Agent 工作流：

```text
请求 → 规划 → RAG 检索 → 可选的运单查询/异常评估 → 审批门 → 答复
                                                       └→ 本地 DEMO 工单
```

## 快速开始

已测试：Windows、Python 3.12、本机 Ollama。建议 Python 3.11 或 3.12。

1. 安装并启动 [Ollama](https://ollama.com/)，下载模型：

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

如果 Ollama 没有运行，在另一个终端执行 `ollama serve`。模型下载需要网络和数 GB 磁盘空间，推理速度取决于机器配置。

2. 克隆项目，创建环境：

```bash
git clone https://github.com/eligiblex/logistics-rag-assistant.git
cd logistics-rag-assistant
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

macOS / Linux：

```bash
source .venv/bin/activate
cp .env.example .env
```

3. 安装并启动：

```bash
python -m pip install -r requirements.txt
python main.py
```

打开 [本地页面](http://127.0.0.1:8000) 或 [接口文档](http://127.0.0.1:8000/docs)。首次启动会生成索引；缺少模型、资料损坏或 Ollama 未启动会导致启动失败，应先检查终端日志。

## 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | /api/health | 初始化就绪检查，不主动探测模型存活 |
| GET | /api/status | 模型、运行时间、刷新状态 |
| POST | /api/query | 普通问答，返回 answer / sources / elapsed_ms |
| POST | /api/query/stream | SSE：sources → token → done；失败为 error |
| GET | /api/knowledge-base/info | 片段数量、Embedding 模型、刷新状态 |
| POST | /api/knowledge-base/refresh | 后台刷新，接受返回 202，已有任务返回 409 |
| POST | /api/agent/runs | 创建一次 Agent 运行 |
| GET | /api/agent/runs/{run_id} | 查询状态、工具结果和 Trace |
| POST | /api/agent/runs/{run_id}/decision | 批准或拒绝待执行动作 |

查询请求：

```json
{"question": "运单应包含哪些信息？"}
```

流式事件示例（示意结构，不是性能结果）：

```text
data: {"type":"sources","sources":[...]}

data: {"type":"token","content":"运单应包含"}

data: {"type":"done","elapsed_ms":1234}
```

## 配置与资料更新

配置优先级：系统环境变量 > 项目根目录 .env > config.yaml > 默认值。文件路径相对项目根目录解析。

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| LLM_MODEL | qwen2.5:7b | Ollama 已下载的生成模型 |
| EMBEDDING_MODEL | nomic-embed-text | 向量模型 |
| OLLAMA_BASE_URL | http://127.0.0.1:11434 | 改为远程地址会发送问题和资料给该服务 |
| OLLAMA_TIMEOUT | 120 | 单次网络请求超时，秒 |
| OLLAMA_TRUST_ENV | false | 是否让 Ollama 客户端读取系统代理；本机服务建议关闭，避免 localhost 被代理成 502 |
| CHROMA_DIRECTORY | ./chroma_db | 索引目录 |
| KNOWLEDGE_DATA_DIR | ./data | 资料目录 |
| SERVER_HOST / SERVER_PORT | 127.0.0.1 / 8000 | 默认只监听本机 |

config.yaml 还可调整 chunk_size、chunk_overlap、embedding_batch_size、k、score_threshold。`embedding_batch_size` 默认设为 1，以兼容本地内存较小或不支持大批量请求的 Ollama；机器资源充足时可逐步调大。阈值是向量相关性分数，不是答案正确率；未校准，不保证拒答所有无关问题。修改 Embedding 模型请使用新的索引目录，避免混用向量。

把合法持有、无隐私的资料放进 data 后刷新。扫描 PDF 不含文本层时会被跳过，本项目未集成 OCR。空资料不会覆盖已有索引。

## 测试与检索检查

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
node tests/test_sse.cjs
```

单元测试不需要 Ollama、外网或 API Key，使用确定性假向量和假模型。Agent 测试覆盖工具规划、异常分级、人工批准/拒绝、SQLite 恢复和 API 状态冲突。Node 测试覆盖所有字节切分位置、中文/emoji、换行、CRLF 及中断事件。

真实 Ollama 检索检查（需要已下载 Embedding 模型，不调用生成模型）：

```bash
python scripts/evaluate_retrieval.py
```

小样本检查只用于发现检索回归，不代表线上准确率。实际验证范围见 [验证记录](docs/validation.md)。

## Docker Compose（可选）

```bash
docker compose up --build
```

启动顺序：Ollama 服务 → 模型下载任务 → API；模型和索引用独立卷保存。只向本机映射 8000 端口，默认 CPU 推理。首次下载可能较久，需要 Docker Compose v2。本轮未在 Docker 环境完成端到端验证，不能据此宣称已完成生产部署。

可选 Gradio 入口：

```bash
python -m pip install -r requirements-gradio.txt
python main.py --mode gradio
```

Gradio 是备选展示入口，核心验证以 FastAPI 自定义网页为准。

## 项目结构

```text
src/          配置、文档处理、向量索引、RAG 流水线、Agent、API
static/       前端样式、SSE 解析与聊天交互
templates/    网页模板
data/         教学示例知识文本
tests/        离线测试与前端协议测试
scripts/      小样本检索检查
docs/         验证范围与使用边界
```

建议从 [Agent 学习与断点指南](docs/agent-learning-guide.md) 开始，按文件顺序调试，并用 Swagger 完成一次“暂停 → 人工批准 → 恢复”演示。

## 已知边界

- 默认资料只是小型教学样例，不是企业知识资产；事实、规则和公式需在实际业务中另行核实。
- 引用片段可追踪，不等于引用一定正确；模型仍可能产生幻觉。
- 采用单进程内嵌 Chroma 和线程锁，不支持多实例共享目录部署；刷新跨批次不是事务，失败后应重试同步并检查状态。
- 尚未实现混合检索、Reranker、鉴权、用户隔离、速率限制及正式基准评测。请勿直接暴露公网。
- 当前 Agent 使用显式规则规划，便于学习和测试；尚未接入模型结构化规划、真实订单沙箱、分布式任务队列与多实例共享状态。
- 当前公开展示项目，尚未指定开源许可证。模型和依赖遵循各自的使用条款。
