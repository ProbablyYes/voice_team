from flask import Flask, render_template, request, jsonify
import os
import json
import urllib.request
import subprocess
from pathlib import Path
from werkzeug.utils import secure_filename

from backend.video_generator import generate_video
from backend.model_trainer import train_model
from backend.chat_engine import chat_response

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30MB，防止误传太大


# -------------------------
# 工具函数
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def normalize_ref_audio_path(p: str) -> str:
    """
    允许前端传：
      - /static/audios/tts.wav
      - static/audios/tts.wav
      - 绝对路径 /home/.../tts.wav （不建议，但兼容）
    最终尽量转成磁盘路径给后端推理用。
    """
    if not p:
        return p
    p = p.strip().replace("\\", "/")
    if p.startswith("/static/"):
        return str(Path(app.root_path) / p.lstrip("/"))
    if p.startswith("static/"):
        return str(Path(app.root_path) / p)
    return p


def call_tts_service(text: str, out_abs_path: str, speaker_id: int = 0) -> str:
    payload = json.dumps({
        "text": text,
        "out_path": out_abs_path,
        "speaker_id": int(speaker_id)
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:5003/tts",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if not data.get("ok"):
        raise RuntimeError(f"TTS failed: {data}")

    # TTS 服务可能返回实际写出的 wav 绝对路径
    return data.get("wav_path") or out_abs_path


# -------------------------
# 页面路由
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_generation", methods=["GET", "POST"])
def video_generation():
    if request.method == "POST":
        data = {
            "model_name": request.form.get("model_name"),
            "model_param": request.form.get("model_param"),
            "ref_audio": request.form.get("ref_audio"),
            "gpu_choice": request.form.get("gpu_choice"),
            "target_text": request.form.get("target_text"),
        }

        # 方案A：这里不做 TTS，只负责用 ref_audio 去生成视频
        data["ref_audio"] = normalize_ref_audio_path(data.get("ref_audio", ""))

        try:
            video_path = generate_video(data)

            # 保证前端能访问：尽量返回 /static/... 的URL
            # 如果你的 generate_video 已经返回 /static/...，下面就不会破坏
            video_path = (video_path or "").replace("\\", "/")
            if not video_path.startswith("/"):
                video_path = "/" + video_path

            return jsonify({"status": "success", "video_path": video_path})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    return render_template("video_generation.html")


@app.route("/model_training", methods=["GET", "POST"])
def model_training():
    if request.method == "POST":
        data = {
            "model_choice": request.form.get("model_choice"),
            "ref_video": request.form.get("ref_video"),
            "gpu_choice": request.form.get("gpu_choice"),
            "epoch": request.form.get("epoch"),
            "custom_params": request.form.get("custom_params"),
        }

        video_path = train_model(data)
        video_path = (video_path or "").replace("\\", "/")
        if not video_path.startswith("/"):
            video_path = "/" + video_path

        return jsonify({"status": "success", "video_path": video_path})

    return render_template("model_training.html")


@app.route("/chat_system", methods=["GET", "POST"])
def chat_system():
    if request.method == "POST":
        data = {
            "model_name": request.form.get("model_name"),
            "model_param": request.form.get("model_param"),
            "voice_clone": request.form.get("voice_clone"),
            "api_choice": request.form.get("api_choice"),
        }

        video_path = chat_response(data)
        video_path = (video_path or "").replace("\\", "/")
        if not video_path.startswith("/"):
            video_path = "/" + video_path

        return jsonify({"status": "success", "video_path": video_path})

    return render_template("chat_system.html")


# -------------------------
# 录音上传：修复“webm/ogg 假装 wav”的致命漏洞
# -------------------------
@app.route("/save_audio", methods=["POST"])
def save_audio():
    if "audio" not in request.files:
        return jsonify({"status": "error", "message": "没有音频文件"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"status": "error", "message": "没有选择文件"}), 400

    out_dir = Path(app.root_path) / "static" / "audios"
    ensure_dir(out_dir)

    filename = secure_filename(audio_file.filename)
    ext = Path(filename).suffix.lower() or ".webm"

    raw_path = out_dir / f"input_raw{ext}"
    audio_file.save(raw_path)

    # 转成真正 input.wav（需要服务器安装 ffmpeg）
    wav_path = out_dir / "input.wav"
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_path),
            "-ac", "1",
            "-ar", "16000",
            str(wav_path)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        return jsonify({"status": "error", "message": f"转码失败：{e}"}), 500

    return jsonify({"status": "success", "message": "音频保存成功", "wav_url": "/static/audios/input.wav"})


@app.route("/audio_exists", methods=["GET"])
def audio_exists():
    p = Path(app.root_path) / "static" / "audios" / "input.wav"
    return jsonify({"exists": p.exists(), "url": "/static/audios/input.wav" if p.exists() else ""})


# -------------------------
# 方案A 的关键：仅做 TTS → 返回给前端（URL + 后端可用路径）
# -------------------------
@app.route("/api/tts", methods=["POST"])
def api_tts():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"status": "error", "error": "empty text"}), 400

    speaker_id = int(body.get("speaker_id", 0))

    # 安全：防路径穿越
    out_name = secure_filename((body.get("out_name") or "tts.wav").strip())
    out_name = Path(out_name).name
    if not out_name.endswith(".wav"):
        out_name += ".wav"

    out_rel = f"static/audios/{out_name}"                 # 给后端用（相对路径）
    out_abs = str(Path(app.root_path) / out_rel)          # 实际落盘绝对路径
    ensure_dir(Path(out_abs).parent)

    try:
        wav_abs = call_tts_service(text=text, out_abs_path=out_abs, speaker_id=speaker_id)

        # 前端播放用 URL（不要给绝对磁盘路径）
        audio_url = "/" + out_rel.replace("\\", "/")

        return jsonify({
            "status": "success",
            "audio_url": audio_url,
            "audio_path": out_rel,   # ⭐ 方案A：前端把它自动回填到 ref_audio
            "wav_abs": wav_abs if False else ""  # 不建议暴露，默认不返回
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# 可选：你 video_generation.html 里有 /tts_test 的链接，就提供一个页面避免 404
@app.route("/tts_test", methods=["GET"])
def tts_test():
    return render_template("tts_test.html")


if __name__ == "__main__":
    app.run(debug=False, port=5001)
