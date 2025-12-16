# 说话人脸生成对话系统

## 系统流程

```
[用户点击“生成视频”按钮]
        ↓
[前端 JS 捕获表单数据并用 fetch 发送 POST 请求]
        ↓
[Flask 路由接收 request.form]
        ↓
[调用 backend/video_generator.py 中的函数 generate_video()]
        ↓
[后端函数返回生成视频的路径]
        ↓
[Flask 把路径以 JSON 形式返回给前端]
        ↓
[前端 JS 接收到路径 → 替换 <video> 标签的 src → 自动播放视频]
```

## 核心模块
- **训练后端**: `./backend/model_trainer.py` - 负责调用模型执行训练任务
- **推理后端**: `./backend/video_generator.py` - 负责调用模型执行视频生成推理

## Demo 使用方法

1. 安装依赖：
   ```bash
   pip install flask
   ```

2. 启动应用：
   ```bash
   python app.py
   ```

3. 访问应用：
   打开 http://127.0.0.1:5001

   进行训练，视频生成，对话等操作。

## GeneFace 环境构建 (Docker)

为了成功构建 GeneFace 的 Docker 镜像（尤其是在 Windows 环境下），请严格按照以下步骤操作：

1.  **准备工作**：
    *   确保已安装 Docker Desktop。
    *   确保 `GeneFace-main` 目录下包含 `.dockerignore` 文件（已创建，用于排除大文件）。
    *   确保 `GeneFace-main/Dockerfile` 已修改为使用 `pip` 镜像源并限制编译并发数（已修改）。
    *   确保 `GeneFace-main/docker/requirements.txt` 已移除严格版本限制（已修改）。
    *   确保 `GeneFace-main/docker/install_ext.sh` 中的路径已修正（已修改）。

2.  **构建镜像**：
    在项目根目录下打开 PowerShell 终端，运行以下命令。
    
    > 注意：我们使用 `DOCKER_BUILDKIT=0` 来使用旧版构建器，以避免 Windows 下 BuildKit 缓存损坏的问题。

    ```powershell
    $env:DOCKER_BUILDKIT=0; cd GeneFace-main; docker build -t geneface:latest -f Dockerfile .
    ```

    *构建过程可能需要较长时间（取决于网络和机器性能），请耐心等待直到出现 `Successfully tagged geneface:latest`。*

3.  **验证构建**：
    构建完成后，可以使用以下命令检查镜像是否存在：
    
    ```powershell
    docker images | findstr geneface
    ```

## Web 平台使用指南

### 1. 启动 Web 服务
在项目根目录下（`voice_team-main`）运行：
```bash
python app.py
```
服务启动后，访问浏览器：[http://127.0.0.1:5001](http://127.0.0.1:5001)

### 2. 模型训练 (GeneFace)
1.  点击导航栏的 **“模型训练”**。
2.  **上传视频**：选择一个包含单人说话的视频文件（建议 1-3 分钟，MP4 格式，文件名尽量用英文）。
3.  **选择设备**：**推荐选择 "GPU"**。
    *   *注意：请确保你的机器拥有 NVIDIA 显卡并正确配置了 CUDA 环境，否则训练速度会非常慢或失败。*
4.  **设置 Epoch**：训练轮数。
5.  点击 **“开始训练”**。
    *   系统会自动调用 Docker 容器进行数据预处理（音频提取、人脸检测、3DMM 提取）和 NeRF 模型训练。

### 3. 视频生成 (推理)
1.  点击导航栏的 **“视频生成”**。
2.  **选择模型**：输入训练好的 Video ID（即上传视频的文件名，例如上传了 `May.mp4`，则 ID 为 `May`）。
3.  **上传音频**：上传一段驱动音频（WAV/MP3）。
4.  **选择设备**：**推荐选择 "GPU"**。
5.  点击 **“生成视频”**。
    *   系统将调用 Docker 容器进行推理，生成的视频将显示在页面上。

