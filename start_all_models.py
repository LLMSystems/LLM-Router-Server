import argparse
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
            cli_args = build_cli_args_from_dict(model_cfg)
            logger.info(f"執行指令: vllm {' '.join(cli_args)}")
            proc = subprocess.Popen(
                ["vllm"] + cli_args,
                start_new_session=True
            )
            running_processes[model_name] = proc
            time.sleep(2)
        except Exception as e:
            logger.error(f"啟動模型 {model_name} 時發生錯誤: {e}")
            
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

            
    
