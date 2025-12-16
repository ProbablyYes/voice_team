import subprocess
import os
import time
import shutil

def train_model(data):
    """
    模拟模型训练逻辑。
    """
    print("[backend.model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")
    
    video_path = data['ref_video']
    print(f"输入视频：{video_path}")

    print("[backend.model_trainer] 模型训练中...")

    if data['model_choice'] == "SyncTalk":
        try:
            # 构建命令
            cmd = [
                "./SyncTalk/run_synctalk.sh", "train",
                "--video_path", data['ref_video'],
                "--gpu", data['gpu_choice'],
                "--epochs", data['epoch']
            ]
            
            print(f"[backend.model_trainer] 执行命令: {' '.join(cmd)}")
            # 执行训练命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print("[backend.model_trainer] 训练输出:", result.stdout)
            if result.stderr:
                print("[backend.model_trainer] 错误输出:", result.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] 训练失败，退出码: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            return video_path
        except FileNotFoundError:
            print("[backend.model_trainer] 错误: 找不到训练脚本")
            return video_path
        except Exception as e:
            print(f"[backend.model_trainer] 训练过程中发生未知错误: {e}")
            return video_path

    elif data['model_choice'] == "GeneFace":
        try:
            # 处理视频路径
            video_path = data['ref_video'].replace('\\', '/')
            if video_path.startswith('/'):
                video_path = video_path[1:]
            
            # 确保视频文件存在并复制到 GeneFace 目录
            geneface_video_dir = os.path.join("GeneFace-main", "data", "raw", "videos")
            os.makedirs(geneface_video_dir, exist_ok=True)
            
            video_filename = os.path.basename(video_path)
            target_video_path = os.path.join(geneface_video_dir, video_filename)
            
            if os.path.exists(video_path):
                if os.path.abspath(video_path) != os.path.abspath(target_video_path):
                    shutil.copy2(video_path, target_video_path)
                    print(f"[backend.model_trainer] 已复制视频到: {target_video_path}")
            
            # 容器内的相对路径
            container_video_path = f"data/raw/videos/{video_filename}"

            # 计算绝对路径用于挂载
            cwd = os.getcwd()
            geneface_dir = os.path.join(cwd, "GeneFace-main")
            geneface_abs = os.path.abspath(geneface_dir)

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
            # docker run --rm {gpu_flag} -v "{geneface_abs}:/GeneFace" -w "/GeneFace" geneface:latest bash scripts/train_pipeline.sh --video_path "{container_video_path}" --epochs "{epochs}"
            
            docker_cmd = ["docker", "run", "--rm"]
            if gpu_flag:
                docker_cmd.extend(gpu_flag.split())
            
            # 添加 PYTHONUNBUFFERED=1 环境变量以禁用 Python 输出缓冲
            docker_cmd.extend(["-e", "PYTHONUNBUFFERED=1"])

            docker_cmd.extend([
                "-v", f"{geneface_abs}:/GeneFace",
                "-w", "/GeneFace",
                "geneface:latest",
                "bash", "scripts/train_pipeline.sh",
                "--video_path", container_video_path,
                "--epochs", str(data['epoch'])
            ])
            
            print(f"[backend.model_trainer] 执行命令: {' '.join(docker_cmd)}")
            
            # 实时读取输出
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # 打印实时日志
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"[GeneFace Docker] {output.strip()}")
            
            rc = process.poll()
            if rc != 0:
                print(f"[backend.model_trainer] 训练失败，退出码: {rc}")
                return video_path
            else:
                print("[backend.model_trainer] 训练成功完成")

        except FileNotFoundError:
            print("[backend.model_trainer] 错误: 找不到训练脚本或Docker未安装")
            return video_path
        except Exception as e:
            print(f"[backend.model_trainer] 训练过程中发生未知错误: {e}")
            return video_path

    print("[backend.model_trainer] 训练完成")
    return video_path
