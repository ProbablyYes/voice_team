# 说话人脸生成对话系统（Docker 部署与复现文档）

## 重要：当前分支的目录结构

本分支已按要求调整为：仓库根目录只保留 `README.md` 与 `GENEFACE/`。

因此，**除非特别说明，本文档所有命令都假设你先进入 `GENEFACE/` 再执行**：

```bash
cd GENEFACE
```

本项目包含三个主要功能（与课程 demo 对齐）：
- **模型训练**：基于 `GeneFace` 训练指定人物的说话人脸模型
- **视频生成/推理**：输入人物模型 + 驱动音频，生成说话视频
- **实时对话**：语音录制 → ASR → LLM → TTS → 视频生成

同时提供（3.2）评测容器：输出 **NIQE / PSNR / SSIM / FID / LSE-C / LSE-D**（其中 LSE 需要额外权重与评测脚本接入，见下文）。

> 重要：助教复现时最容易失败的地方不是代码本身，而是 **路径、端口、Docker/GPU 运行时、模型缓存**。本文档把这些关键点全部显式写出，按顺序执行即可复现。

---

## 0. 仓库结构与真实调用链路（读代码得出的结论）

### 0.1 Web（前端 + Flask）
- **Web 入口**：`app.py`，默认端口 **5001**
- **训练后端**：`backend/model_trainer.py::train_model`
- **推理后端**：`backend/video_generator.py::generate_video`
- **对话后端**：`backend/chat_engine.py::chat_response`

### 0.2 GeneFace（训练/推理的真实执行方式）
本项目并不是“在宿主机直接跑 GeneFace”，而是 **由 Flask 后端在运行时执行 `docker run ... geneface:latest`**：
- **训练容器内执行**：`GeneFace-main/scripts/train_pipeline.sh`
- **推理容器内执行**：`GeneFace-main/scripts/infer_pipeline.sh`
- **宿主机目录挂载到容器**：`-v <abs_path>/GeneFace-main:/GeneFace`
- **容器内输出视频**：`GeneFace-main/infer_out/<video_id>/pred_video/<audio_name>.mp4`
- **Web 展示视频**（复制到静态目录）：`static/videos/geneface_<video_id>_<audio_name>.mp4`

### 0.3 TTS（用于“视频生成页”的文本转音频）
本项目提供独立 TTS 服务：
- **入口**：`tts_server.py`（默认端口 **5003**）
- **Web 调用方式**：`POST /api/tts` → `app.py` 再请求 `http://127.0.0.1:5003/tts`
- **Docker 编排**：`docker-compose.yml`（服务名 `tts`）

> 说明：`chat_engine.py` 的“语音克隆”目前走 `backend/voice_cloner.py` 的 `DummyCloner(edge-tts)`，一般需要外网；而“视频生成页”的 TTS 走 `tts_server.py`（可离线）。如果你的华为云环境限制外网，建议把对话模块也改为调用 `tts_server.py`（接口与 `app.py` 中 `call_tts_service()` 一致）。

---

## 1. 3.1 部署与运行（训练 / 推理 / 实时对话）

### 1.1 前置条件（必读）
- **Docker**：用于运行 GeneFace（训练/推理）与 TTS（可选）
- **GPU（可选但强烈建议）**
  - 需要 NVIDIA 驱动 + `nvidia-container-toolkit`（否则 `docker run --gpus ...` 会失败）
  - 若没有 GPU，训练会非常慢
- **ffmpeg（宿主机）**：用于 `/save_audio` 把 webm/ogg 转为真正的 `input.wav`
- **外网（按你环境而定）**
  - 训练/推理可能会下载模型文件（例如 face_alignment 的 2DFAN4 权重）
  - 本项目会把宿主机 `model_cache/` 挂载到容器 `/root/.cache/torch/hub/checkpoints`，减少重复下载

### 1.2 构建 GeneFace 镜像（`geneface:latest`）
`backend/model_trainer.py` 和 `backend/video_generator.py` 都写死使用镜像名：**`geneface:latest`**，所以必须构建该 tag。

在仓库根目录执行（先 `cd GENEFACE`）：

```bash
docker build -t geneface:latest -f GeneFace-main/Dockerfile GeneFace-main
```

#### 常见问题
- **无法拉取 `nvcr.io/nvidia/pytorch:22.07-py3`**：这是 GeneFace 的基础镜像。若华为云环境无法访问 NGC，可在构建时替换 base image（见 `GeneFace-main/Dockerfile` 的 `ARG BASE_IMAGE=...`）。
- **编译扩展很慢/容易 OOM**：`GeneFace-main/Dockerfile` 里已设置 `MAX_JOBS=1`，仍建议保证足够内存。

### 1.3 启动 TTS（Docker，端口 5003）
在仓库根目录执行（先 `cd GENEFACE`）：

```bash
docker compose up -d tts
```

健康检查：

```bash
curl -fsS http://127.0.0.1:5003/health
```

> 注意：`docker-compose.yml` 使用 `network_mode: "host"`。因此：
> - 不需要 `-p 5003:5003`
> - `app.py` 访问 `127.0.0.1:5003` 才能成功

### 1.4 启动 Web（Flask，端口 5001）
建议用虚拟环境（venv/conda 均可），安装依赖并启动：

```bash
pip install -r requirements.txt
python app.py
```

浏览器访问：`http://127.0.0.1:5001`

---

## 2. 3.1 功能复现步骤（按页面顺序）

### 2.1 模型训练（GeneFace）
页面：`/model_training`

#### 准备数据
- 把单人说话视频（mp4）放到：`static/videos/`
  - 例：`static/videos/May.mp4`
- **页面填写的路径必须是服务器本机路径**（不是 URL）

#### 页面参数（与代码一致）
- **模型选择**：GeneFace
- **参考视频路径**：`static/videos/May.mp4`
- **GPU 选择**：`GPU0`（无 GPU 时不建议；如必须用 CPU，需要你自行把页面与后端逻辑对齐）
- **Epoch**：例如 10

#### 后端真实执行（便于定位输出/报错）
1. 复制视频到：`GeneFace-main/data/raw/videos/<video_filename>`
2. 容器内执行：
   - `bash scripts/train_pipeline.sh --video_path data/raw/videos/<video_filename> --epochs <epoch>`
3. 自动创建配置：`GeneFace-main/egs/datasets/videos/<video_id>/`
   - 从 `egs/datasets/videos/May` 复制 yaml，并把其中的 `May` 替换为 `<video_id>`
4. 如果预处理产物已存在会跳过预处理：
   - `GeneFace-main/data/binary/videos/<video_id>/trainval_dataset.npy`

#### 训练输出
- checkpoints：`GeneFace-main/checkpoints/<video_id>/...`

### 2.2 视频生成（GeneFace 推理）
页面：`/video_generation`

本页面实现了“方案A”：**先 TTS → 再推理**（更适合服务器环境）。

#### 步骤 A：仅测试 TTS 生成 wav
1. 在“目标文本”输入中文
2. 点击“仅测试 TTS 生成 wav”
3. 成功后页面会自动回填 `ref_audio = static/audios/tts.wav`

#### 步骤 B：开始生成视频
- **模型类型**：GeneFace
- **模型ID（video_id）**：训练用的视频文件名去后缀，例如 `May`
- **参考音频路径（ref_audio）**：`static/audios/tts.wav`
- 点击“开始生成视频”

#### 推理输出位置（非常关键）
容器内输出（挂载到宿主机 `GeneFace-main/`）：
- `GeneFace-main/infer_out/<video_id>/pred_video/<audio_name>.mp4`

Web 展示的输出（复制到静态目录）：
- `static/videos/geneface_<video_id>_<audio_name>.mp4`

### 2.3 实时对话（ASR → LLM → TTS → Video）
页面：`/chat_system`

#### 环境变量（必设）
`backend/chat_engine.py` 强依赖智谱 API：
- `ZHIPU_API_KEY`：你的 API Key
- `ZHIPU_MODEL`：可选，默认 `glm-4-flashx`

示例：

```bash
export ZHIPU_API_KEY="xxxx"
export ZHIPU_MODEL="glm-4-flashx"
```

#### ASR 注意事项（常见失败点）
当前 ASR 使用 `speech_recognition` 的 `recognize_google`，需要外网。若华为云无外网，需要替换为离线 ASR（例如 Vosk/Whisper），否则会报 `ASR 服务错误（Google）`。

#### 语音克隆注意事项（华为云“完整运行”的关键）
`backend/voice_cloner.py` 目前：
- `OpenVoiceCloner`：未实现（会抛 `NotImplementedError`）
- 默认回退 `DummyCloner`：使用 `edge_tts`（一般需要外网）

建议改造方案：
- **复用已 docker 化的 `tts_server.py`**（接口 `/tts`），与视频生成页一致
- 改造点：在 `voice_cloner.py` 增加一个 `HttpTTSCloner`，调用 `http://127.0.0.1:5003/tts` 生成 `response_*.wav`

---

## 3. 3.2 模型效果评价（Docker 输出指标）

### 3.2.1 评测模块说明
评测代码：`eval_metrics/evaluate.py`  
评测镜像：`docker/eval/Dockerfile`

当前实现/输出字段：
- **NIQE**：已实现
- **PSNR**：已实现
- **SSIM**：已实现
- **FID**：已实现（但首次可能需要下载 Inception 权重；失败会返回 `null`）
- **LSE-C / LSE-D**：预留接口（默认 `null`，见 3.2.4）

### 3.2.2 构建评测镜像
在仓库根目录执行（先 `cd GENEFACE`）：

```bash
docker build -t talkingface-eval:latest -f docker/eval/Dockerfile .
```

### 3.2.3 运行评测（输出指标）
评测镜像入口为 `python -m eval_metrics.evaluate`，你只需要把宿主机目录挂载进容器：

```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  talkingface-eval:latest \
  --gt_video "/workspace/<GT视频路径>.mp4" \
  --pred_video "/workspace/<生成视频路径>.mp4" \
  --out_json "/workspace/metrics.json"
```

程序会在 stdout 打印 JSON，同时写出 `metrics.json`（如果你传了 `--out_json`）。

### 3.2.4 LSE-C / LSE-D 接入（必须读）
课程的 LSE 指标通常依赖 **SyncNet/Wav2Lip** 类口型同步网络（需要预训练权重 + 特定评测脚本）。

本仓库在 `eval_metrics/evaluate.py` 预留了接入点：
- 函数：`_maybe_compute_lse(...)`

你们只需要把课程提供的 SyncNet 权重与评测脚本放进仓库（例如 `eval_metrics/lse/`），然后在 `_maybe_compute_lse` 内调用并返回 `(lse_c, lse_d)` 即可。

---

## 4. 3.3 助教复现清单（推荐顺序）

### 4.1 一次性环境检查
- Docker：`docker ps`
- GPU（若使用）：`docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`
- ffmpeg：`ffmpeg -version`

### 4.2 启动顺序（最稳）
1. 构建 GeneFace：`geneface:latest`
2. 启动 TTS：`docker compose up -d tts`，并 `curl /health`
3. 启动 Web：`python app.py`
4. 页面验证：
   - **模型训练**：产生 `GeneFace-main/checkpoints/<video_id>/...`
   - **视频生成**：产生 `static/videos/geneface_*.mp4`
   - **实时对话**：设置 `ZHIPU_API_KEY`；确认 ASR/TTS 的外网/离线条件
5. 评测（3.2）：用 `talkingface-eval:latest` 对 GT 与生成视频计算指标

---

## 5. 常见错误与定位

### 5.1 `docker ... capabilities: [[gpu]]`
- 原因：未安装 `nvidia-container-toolkit` 或宿主机无 NVIDIA 驱动
- 解决：安装 GPU 运行时，或改为 CPU（训练耗时会显著增加）

### 5.2 推理结束但 Web 显示 `out.mp4`
- 原因：后端按固定路径找输出：
  - `GeneFace-main/infer_out/<video_id>/pred_video/<audio_name>.mp4`
- 解决：检查 `<video_id>` 与 `<audio_name>` 是否与输入一致（音频文件名会影响输出文件名）

### 5.3 `/save_audio` 转码失败
- 原因：宿主机缺少 ffmpeg
- 解决：安装 ffmpeg（或自行把转码逻辑迁到容器）

