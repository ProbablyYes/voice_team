import os
import re
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file

# ---- 强制离线 ----
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from melo.api import TTS

app = Flask(__name__)

DEVICE = os.environ.get("TTS_DEVICE", "cpu")
DEFAULT_SPK = int(os.environ.get("TTS_SPEAKER_ID", "0"))

# 只保留中文 + 常见中文标点 + 空白
ZH_KEEP_RE = re.compile(r"[^\u4e00-\u9fff，。！？；：、“”‘’（）《》【】—…\s]+")

MODEL = None

def sanitize_zh(text: str) -> str:
    text = (text or "").strip()
    text = ZH_KEEP_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def patch_language_map():
    """
    关键：把 ZH_MIX_EN 也映射到中文模块，避免 cleaner KeyError。
    同时禁用其它语言分支（你只要中文）。
    """
    import melo.text.cleaner as cleaner
    from melo.text import chinese as zh_mod

    # 确保这两个 key 都存在
    cleaner.language_module_map["ZH"] = zh_mod
    cleaner.language_module_map["ZH_MIX_EN"] = zh_mod

    # 可选：把其它语言 key 移除（更“纯中文”）
    for k in list(cleaner.language_module_map.keys()):
        if k not in ("ZH", "ZH_MIX_EN"):
            cleaner.language_module_map.pop(k, None)

def _get_hps_bert_dim(model) -> int:
    hps = getattr(model, "hps", None)
    for a, b in [("data", "bert_dim"), ("model", "bert_dim")]:
        try:
            obj = getattr(hps, a, None)
            v = getattr(obj, b, None) if obj is not None else None
            if isinstance(v, int) and v > 0:
                return v
        except Exception:
            pass
    return 768

def patch_no_bert(model):
    """
    关键：把 melo 的 get_bert 替换成全0特征，彻底不走 transformers/huggingface。
    注意：melo.utils 里是 from melo.text import get_bert 导入的，所以两边都要 patch。
    """
    import melo.text as mtext
    import melo.utils as mutils

    bert_dim = _get_hps_bert_dim(model)

    def dummy_get_bert(norm_text, word2ph, language_str, device):
        try:
            phone_len = int(sum(word2ph)) if word2ph is not None else 1
        except Exception:
            phone_len = 1
        phone_len = max(phone_len, 1)
        return torch.zeros((bert_dim, phone_len), dtype=torch.float32, device=device)

    mtext.get_bert = dummy_get_bert
    mutils.get_bert = dummy_get_bert

@app.get("/health")
def health():
    return jsonify({"ok": True, "device": DEVICE, "offline": True, "no_bert": True, "zh_only": True})

def get_model():
    global MODEL
    if MODEL is None:
        patch_language_map()
        MODEL = TTS(language="ZH", device=DEVICE)
        patch_no_bert(MODEL)
    return MODEL

@app.post("/tts")
def tts():
    try:
        payload = request.get_json(force=True)
        text = payload.get("text", "")
        out_path = payload.get("out_path", "")
        speaker_id = int(payload.get("speaker_id", DEFAULT_SPK))
        speed = float(payload.get("speed", 1.0))

        text2 = sanitize_zh(text)
        if not text2:
            return jsonify({"ok": False, "error": "输入文本剔除非中文后为空，请只输入中文。"}), 400

        if not out_path:
            out_path = "/root/voice_team/static/audios/tts.wav"

        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        model = get_model()
        model.tts_to_file(text2, speaker_id=speaker_id, output_path=str(out_p), speed=speed)

        return jsonify({"ok": True, "wav_path": str(out_p), "text_used": text2})

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print("[TTS ERROR]", err)
        traceback.print_exc()
        return jsonify({"ok": False, "error": err}), 500

@app.get("/download")
def download():
    p = request.args.get("path", "")
    if not p:
        return jsonify({"ok": False, "error": "missing path"}), 400
    p = Path(p)
    if not p.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_file(str(p), as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False)
