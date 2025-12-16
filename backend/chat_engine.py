import os
import speech_recognition as sr
from zhipuai import ZhipuAI
from backend.voice_cloner import get_voice_cloner
from backend.video_generator import generate_video

def chat_response(data):
    """
    模拟实时对话系统视频生成逻辑。
    """
    print("[backend.chat_engine] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    # 1. 语音转文字
    # input_audio = "./static/audios/input.wav"
    # 优先使用前端上传的 input.wav，如果不存在则回退到测试音频
    if os.path.exists("./static/audios/input.wav"):
        input_audio = "./static/audios/input.wav"
    else:
        input_audio = "./SyncTalk/audio/aud.wav"
        
    input_text = "./static/text/input.txt"
    audio_to_text(input_audio, input_text)

    # 2. 大模型回答
    output_text = "./static/text/output.txt"
    api_key = "31af4e1567ad48f49b6d7b914b4145fb.MDVLvMiePGYLRJ7M"
    model = "glm-4-plus"
    ai_response_text = get_ai_response(input_text, output_text, api_key, model)

    # 3. 语音克隆 (TTS)
    print(f"[backend.chat_engine] 开始语音克隆，使用模型: {data.get('voice_clone')}")
    cloner = get_voice_cloner(data.get('voice_clone', 'dummy'))
    
    response_audio_path = "./static/audios/response.wav"
    os.makedirs(os.path.dirname(response_audio_path), exist_ok=True)
    
    # 使用输入音频作为音色参考
    cloner.clone_voice(
        text=ai_response_text,
        ref_audio_path=input_audio,
        output_path=response_audio_path
    )

    # 4. 视频生成
    print(f"[backend.chat_engine] 开始生成视频，使用模型: {data.get('model_name')}")
    video_gen_data = {
        "model_name": data.get('model_name'),
        "model_param": data.get('model_param'),
        "ref_audio": response_audio_path,
        "gpu_choice": "GPU0" # 默认使用 GPU0
    }
    
    video_path = generate_video(video_gen_data)
    
    print(f"[backend.chat_engine] 流程完成，视频路径：{video_path}")
    return video_path

def audio_to_text(input_audio, input_text):
    try:
        # 初始化识别器
        recognizer = sr.Recognizer()
        
        # 加载音频文件
        with sr.AudioFile(input_audio) as source:
            # 调整环境噪声
            recognizer.adjust_for_ambient_noise(source)
            # 读取音频数据
            audio_data = recognizer.record(source)
            
            print("正在识别语音...")
            
            # 使用Google语音识别
            text = recognizer.recognize_google(audio_data, language='zh-CN')
            
            # 将结果写入文件
            with open(input_text, 'w', encoding='utf-8') as f:
                f.write(text)
                
            print(f"语音识别完成！结果已保存到: {input_text}")
            print(f"识别结果: {text}")
            
            return text
            
    except sr.UnknownValueError:
        print("无法识别音频内容")
    except sr.RequestError as e:
        print(f"语音识别服务错误: {e}")
    except FileNotFoundError:
        print(f"音频文件不存在: {input_audio}")
    except Exception as e:
        print(f"发生错误: {e}")

def get_ai_response(input_text, output_text, api_key, model):
    client = ZhipuAI(api_key = api_key)
    with open(input_text, 'r', encoding='utf-8') as file:
        content = file.read().strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}]
    )
    output = response.choices[0].message.content

    with open(output_text, 'w', encoding='utf-8') as file:
        file.write(output)

    print(f"答复已保存到: {output_text}")
    return output