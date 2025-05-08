import argparse
import copy
import os
import signal
import subprocess
import sys
import time
from typing import Dict

from vllm.logger import init_logger

from app.config_loader import load_config
from app.env import env_setup
from app.vllm_launcher import build_cli_args_from_dict

logger = init_logger(__name__)

running_processes: Dict[str, subprocess.Popen] = {}

def launch_all_models(config_path):
    env_setup()
    config = load_config(path=config_path)
    engines = config.get("LLM_engines", {})

    for model_name, model_cfg in engines.items():
        try:
            model_cfg_cleaned = copy.deepcopy(model_cfg)
            if model_cfg_cleaned.get("tensor_parallel_size", 1) == 1:
                cuda_id = model_cfg_cleaned.pop("cuda_device", None)
                
            cli_args = build_cli_args_from_dict(model_cfg_cleaned)
            logger.info(f"執行指令: vllm {' '.join(cli_args)}")
            cuda_env = os.environ.copy()
            if cuda_id is not None:
                cuda_env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
                logger.info(f"設定 {model_name} 使用 GPU {cuda_id}")
                
            proc = subprocess.Popen(
                ["vllm"] + cli_args,
                env=cuda_env,
                start_new_session=True
            )
            running_processes[model_name] = proc
            time.sleep(2)
        except Exception as e:
            logger.error(f"啟動模型 {model_name} 時發生錯誤: {e}")
            
    # embedding server
    embedding_server_cfg = config.get("embedding_server", {})
    if embedding_server_cfg:
        try:
            logger.info("啟動 Embedding / Reranker Server ...")

            cuda_env = os.environ.copy()
            cuda_device = embedding_server_cfg.get("cuda_device")
            if cuda_device is not None:
                cuda_env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
                logger.info(f"設定 Embedding Server 使用 GPU {cuda_device}")

            proc = subprocess.Popen([
                sys.executable, "-m", "embedding_reranker_server.embedding_reranker_launcher",
                "--config", config_path
            ], env=cuda_env, start_new_session=True)

            running_processes["embedding_server"] = proc
        except Exception as e:
            logger.error(f"啟動 Embedding Server 失敗: {e}")
            
def shutdown_all_models():
    logger.info("關閉所有模型...")
    for model_name, proc in running_processes.items():
        try:
            logger.info(f"   → 正在關閉 {model_name} (PID={proc.pid})")
            proc.terminate()
            proc.wait(timeout=5)
        except Exception as e:
            logger.error(f"   關閉 {model_name} 失敗: {e}")
    sys.exit(0)
    
    
def main(config_path):
    signal.signal(signal.SIGINT, lambda sig, frame: shutdown_all_models())
    signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_all_models())

    launch_all_models(config_path)

    while True:
        time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    args = parser.parse_args()

    config_path = args.config
    main(config_path)

            
    
