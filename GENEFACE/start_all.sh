#!/usr/bin/env bash
set -euo pipefail

# ====== 你需要确认的路径 ======
APP_DIR="/root/voice_team"                 # 你的 app.py 所在目录
APP_FILE="${APP_DIR}/app.py"

TTS_ENV="openvoice_env"                    # 运行 TTS 的 conda env（py3.9+）
APP_ENV="voice_env"                        # 运行 Flask 的 conda env（py3.8）

# 你的 TTS 服务脚本（就是我之前给你的 tts_server.py）
TTS_SERVER="/root/voice_team/tts_server.py"

APP_PORT=5001
TTS_PORT=5003

LOG_DIR="${APP_DIR}/logs"
mkdir -p "${LOG_DIR}"

# conda 主程序路径（按你机器上的 miniconda 位置）
CONDA="/root/miniconda3/bin/conda"

echo "[0] Checking files..."
test -f "${APP_FILE}" || { echo "ERROR: app.py not found: ${APP_FILE}"; exit 1; }
test -f "${TTS_SERVER}" || { echo "ERROR: tts_server.py not found: ${TTS_SERVER}"; exit 1; }

echo "[1] Kill old processes on ports ${APP_PORT}/${TTS_PORT} if any..."
# 使用 lsof 找端口占用进程并杀掉
if command -v lsof >/dev/null 2>&1; then
  for p in "${APP_PORT}" "${TTS_PORT}"; do
    PID=$(lsof -tiTCP:${p} -sTCP:LISTEN || true)
    if [ -n "${PID}" ]; then
      echo " - killing PID ${PID} on port ${p}"
      kill -9 ${PID} || true
    fi
  done
else
  echo "WARN: lsof not found. Installing..."
  apt-get update -y && apt-get install -y lsof
fi

echo "[2] Start TTS service (${TTS_ENV}) on 127.0.0.1:${TTS_PORT} ..."
nohup "${CONDA}" run -n "${TTS_ENV}" python "${TTS_SERVER}" \
  > "${LOG_DIR}/tts_${TTS_PORT}.log" 2>&1 &

sleep 1

echo "[3] Start Flask app (${APP_ENV}) on port ${APP_PORT} ..."
# 注意：你 app.py 里 app.run(debug=True) 会开启 reloader，可能产生两个进程
# 所以建议你把 debug=False 或 use_reloader=False（否则 nohup 里日志会乱）
nohup "${CONDA}" run -n "${APP_ENV}" python "${APP_FILE}" \
  > "${LOG_DIR}/app_${APP_PORT}.log" 2>&1 &

echo "[4] Done."
echo " - TTS log: ${LOG_DIR}/tts_${TTS_PORT}.log"
echo " - APP log: ${LOG_DIR}/app_${APP_PORT}.log"
echo "Check ports:"
echo "  lsof -iTCP:${TTS_PORT} -sTCP:LISTEN"
echo "  lsof -iTCP:${APP_PORT} -sTCP:LISTEN"
