import os
import uuid
import speech_recognition as sr
from zhipuai import ZhipuAI

from backend.voice_cloner import get_voice_cloner
from backend.video_generator import generate_video


def chat_response(data):
    """
    实时对话：ASR -> LLM -> Voice Clone(TTS) -> Video
    """
    print("[backend.chat_engine] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    # ---- 路径准备（避免并发覆盖）----
    os.makedirs("./static/audios", exist_ok=True)
    os.makedirs("./static/text", exist_ok=True)

    req_id = uuid.uuid4().hex[:8]
    input_text_path = f"./static/text/input_{req_id}.txt"
    output_text_path = f"./static/text/output_{req_id}.txt"
    response_audio_path = f"./static/audios/response_{req_id}.wav"

    # 1) 选择输入音频
    if os.path.exists("./static/audios/input.wav"):
        input_audio = "./static/audios/input.wav"
    else:
        input_audio = "./SyncTalk/audio/aud.wav"

    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"输入音频不存在: {input_audio}")

    # 2) ASR
    text = audio_to_text(input_audio, input_text_path)
    if not text:
        raise RuntimeError("ASR 失败：未识别到文本")

    # 3) LLM
    api_key = os.getenv("ZHIPU_API_KEY")  # ✅ 环境变量
    if not api_key:
        raise RuntimeError("未设置环境变量 ZHIPU_API_KEY")

    model = os.getenv("ZHIPU_MODEL", "glm-4-flashx")  # ✅ 默认用你想要的
    ai_response_text = get_ai_response(input_text_path, output_text_path, api_key, model)
    if not ai_response_text:
        raise RuntimeError("LLM 返回为空")

    # 4) 语音克隆 / TTS
    print(f"[backend.chat_engine] 开始语音克隆，使用模型: {data.get('voice_clone')}")
    cloner = get_voice_cloner(data.get("voice_clone", "dummy"))

    cloner.clone_voice(
        text=ai_response_text,
        ref_audio_path=input_audio,
        output_path=response_audio_path
    )

    if not os.path.exists(response_audio_path):
        raise RuntimeError("语音克隆失败：未生成 response wav")

    # 5) 视频生成
    print(f"[backend.chat_engine] 开始生成视频，使用模型: {data.get('model_name')}")

    video_gen_data = {
        "model_name": data.get("model_name"),
        "model_param": data.get("model_param"),
        "ref_audio": response_audio_path,
        "gpu_choice": data.get("gpu_choice", "GPU0"),  # ✅ 不要写死
        # "target_text": None  # 如果你的 generate_video 支持可选字段，可显式传
    }

    video_path = generate_video(video_gen_data)
    if not video_path:
        raise RuntimeError("视频生成失败：video_path 为空")

    print(f"[backend.chat_engine] 流程完成，视频路径：{video_path}")
    return video_path


def audio_to_text(input_audio, input_text_path):
    """
    用 speech_recognition 读本地音频文件做识别
    注意：sr.AudioFile 只支持 PCM WAV/AIFF/FLAC
    """
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(input_audio) as source:
            # ✅ 文件识别一般不要 adjust_for_ambient_noise（会吃掉开头）
            audio_data = recognizer.record(source)

        print("[backend.chat_engine] 正在识别语音...")
        text = recognizer.recognize_google(audio_data, language="zh-CN")

        with open(input_text_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[backend.chat_engine] ASR 完成：{text}")
        return text

    except sr.UnknownValueError:
        print("[backend.chat_engine] ASR 无法识别音频内容")
        return ""
    except sr.RequestError as e:
        # Google ASR 需要外网，服务器上很可能失败
        raise RuntimeError(f"ASR 服务错误（Google）：{e}")
    except Exception as e:
        raise RuntimeError(f"ASR 发生错误：{e}")


def get_ai_response(input_text_path, output_text_path, api_key, model):
    client = ZhipuAI(api_key=api_key)

    with open(input_text_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise RuntimeError("LLM 输入为空（ASR 输出为空或文件未写入）")

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}]
    )

    output = resp.choices[0].message.content or ""
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"[backend.chat_engine] LLM 答复已保存到: {output_text_path}")
    return output
