import os
import time
import subprocess
import shutil


def _resolve_host_audio_path(ref_audio: str, project_cwd: str) -> str:
    p = (ref_audio or "").strip().replace('\\', '/')
    if not p:
        raise ValueError("ref_audio 为空，请在页面填写音频路径或先用 TTS 生成 wav")

    if p.startswith("http://") or p.startswith("https://"):
        raise ValueError(f"ref_audio 不能是 URL：{p}，需要填写服务器本机文件路径")

    candidates = []
    # 1) 网页常见写法：/static/audios/xxx.wav
    if p.startswith("/static/"):
        candidates.append(os.path.join(project_cwd, p.lstrip('/')))
    # 2) 相对路径：static/audios/xxx.wav
    if p.startswith("static/"):
        candidates.append(os.path.join(project_cwd, p))
    # 3) 用户可能直接填容器内相对路径：data/raw/val_wavs/xxx.wav
    if p.startswith("data/raw/val_wavs/"):
        candidates.append(os.path.join(project_cwd, "GeneFace-main", p))

    # 4) 绝对路径/相对路径兜底
    if os.path.isabs(p):
        candidates.append(p)
    candidates.append(os.path.join(project_cwd, p))

    # 5) 只填文件名时，默认去 static/audios/ 找
    base = os.path.basename(p)
    if base:
        candidates.append(os.path.join(project_cwd, "static", "audios", base))

    for c in candidates:
        if c and os.path.exists(c):
            return c

    raise FileNotFoundError(
        "找不到 ref_audio 对应的本机文件。"
        f" 输入值: {p}。尝试过: {candidates[:4]}..."
    )

def generate_video(data):
    """
    模拟视频生成逻辑：接收来自前端的参数，并返回一个视频路径。
    """
    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    if data['model_name'] == "SyncTalk":
        try:
            
            # 构建命令
            cmd = [
                './SyncTalk/run_synctalk.sh', 'infer',
                '--model_dir', data['model_param'],
                '--audio_path', data['ref_audio'],
                '--gpu', data['gpu_choice']
            ]

            print(f"[backend.video_generator] 执行命令: {' '.join(cmd)}")

            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # check=True
            )
            
            print("命令标准输出:", result.stdout)
            if result.stderr:
                print("命令标准错误:", result.stderr)
            
            # 文件原路径与目的路径 
            model_dir_name = os.path.basename(data['model_param'])
            source_path = os.path.join("SyncTalk", "model", model_dir_name, "results", "test_audio.mp4")
            audio_name = os.path.splitext(os.path.basename(data['ref_audio']))[0]
            video_filename = f"{model_dir_name}_{audio_name}.mp4"
            destination_path = os.path.join("static", "videos", video_filename)
            # 检查文件是否存在
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                print(f"[backend.video_generator] 视频生成完成，路径：{destination_path}")
                return destination_path
            else:
                print(f"[backend.video_generator] 视频文件不存在: {source_path}")
                # 尝试查找任何新生成的mp4文件
                results_dir = os.path.join("SyncTalk", "model", model_dir_name, "results")
                if os.path.exists(results_dir):
                    mp4_files = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
                    if mp4_files:
                        latest_file = max(mp4_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
                        source_path = os.path.join(results_dir, latest_file)
                        shutil.copy(source_path, destination_path)
                        print(f"[backend.video_generator] 找到最新视频文件: {destination_path}")
                        return destination_path
                
                return os.path.join("static", "videos", "out.mp4")
            
        except subprocess.CalledProcessError as e:
            print(f"[backend.video_generator] 命令执行失败: {e}")
            print("错误输出:", e.stderr)
            return os.path.join("static", "videos", "out.mp4")
        except Exception as e:
            print(f"[backend.video_generator] 其他错误: {e}")
            return os.path.join("static", "videos", "out.mp4")

    elif data['model_name'] == "GeneFace":
        try:
            cwd = os.getcwd()
            host_audio_path = _resolve_host_audio_path(data.get('ref_audio'), cwd)
            
            # 确保音频文件存在并复制到 GeneFace 目录
            geneface_audio_dir = os.path.join("GeneFace-main", "data", "raw", "val_wavs")
            os.makedirs(geneface_audio_dir, exist_ok=True)
            
            audio_filename = os.path.basename(host_audio_path)
            target_audio_path = os.path.join(geneface_audio_dir, audio_filename)

            if os.path.abspath(host_audio_path) != os.path.abspath(target_audio_path):
                shutil.copy2(host_audio_path, target_audio_path)
                print(f"[backend.video_generator] 已复制音频到: {target_audio_path}")

            if not os.path.exists(target_audio_path):
                raise RuntimeError(f"音频复制失败或文件不存在: {target_audio_path}")
            
            # 容器内的相对路径
            container_audio_path = f"data/raw/val_wavs/{audio_filename}"

            # 计算绝对路径用于挂载
            geneface_dir = os.path.join(cwd, "GeneFace-main")
            geneface_abs = os.path.abspath(geneface_dir)

            # 复用宿主机缓存，避免 torch.hub/face_alignment 反复联网下载
            # face_alignment (python-fan) 默认会在 /root/.cache/torch/hub/checkpoints 下找 2DFAN4-*.zip
            model_cache_abs = os.path.abspath(os.path.join(cwd, "model_cache"))
            os.makedirs(model_cache_abs, exist_ok=True)

            # 处理 GPU 参数
            gpu_choice = data.get('gpu_choice', 'GPU0')
            gpu_flag = "--gpus device=0" # 默认
            if gpu_choice.upper() == "CPU":
                gpu_flag = ""
            elif gpu_choice.upper().startswith("GPU"):
                 try:
                     device_id = int(gpu_choice[3:])
                     gpu_flag = f"--gpus device={device_id}"
                 except:
                     pass

            # 构建 Docker 命令
            
            docker_cmd = ["docker", "run", "--rm"]
            if gpu_flag:
                docker_cmd.extend(gpu_flag.split())

            # 确保容器内能 import 顶层包（如 utils/）
            docker_cmd.extend(["-e", "PYTHONPATH=/GeneFace"])
            
            docker_cmd.extend([
                "-v", f"{geneface_abs}:/GeneFace",
                "-v", f"{model_cache_abs}:/root/.cache/torch/hub/checkpoints",
                "-w", "/GeneFace",
                "geneface:latest",
                "bash", "scripts/infer_pipeline.sh",
                "--video_id", data['model_param'],
                "--audio_path", container_audio_path
            ])
            
            print(f"[backend.video_generator] 执行命令: {' '.join(docker_cmd)}")
            
            # 执行命令
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            print("命令标准输出:", result.stdout)
            if result.stderr:
                print("命令标准错误:", result.stderr)

            if result.returncode != 0:
                raise RuntimeError(
                    "GeneFace docker 推理失败（非 0 退出码）。"
                    f"\nstdout: {result.stdout[-4000:]}"
                    f"\nstderr: {result.stderr[-4000:]}"
                )
            
            # 文件原路径与目的路径
            video_id = data['model_param']
            audio_name = os.path.splitext(audio_filename)[0]
            
            source_path = os.path.join("GeneFace-main", "infer_out", video_id, "pred_video", f"{audio_name}.mp4")
            
            video_filename = f"geneface_{video_id}_{audio_name}.mp4"
            destination_path = os.path.join("static", "videos", video_filename)
            
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                print(f"[backend.video_generator] 视频生成完成，路径：{destination_path}")
                return destination_path
            else:
                print(f"[backend.video_generator] 视频文件不存在: {source_path}")
                return os.path.join("static", "videos", "out.mp4")
                
        except Exception as e:
            print(f"[backend.video_generator] GeneFace 推理错误: {e}")
            return os.path.join("static", "videos", "out.mp4")
    
    video_path = os.path.join("static", "videos", "out.mp4")
    print(f"[backend.video_generator] 视频生成完成，路径：{video_path}")
    return video_path
