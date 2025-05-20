# LLM-Router-Server

**2024/09 版本過於老舊，擴展不易，更新困難，因此著手開發易維護、高效、生產級 LLM Server (支援其餘 NLP 模型)**

### 📅 2025/03–04

-   🔧 **思考**  
    採用「路由狀態管理」架構：模型端與 Server 隔離、支援多模型與推理架構迭代

---

### 📅 2025/05/06

-   🔧 **開發**  
    建立 LLM 路由伺服器基本架構：支援配置加載、模型啟動、API 路由、Docker 容器整合

---

### 📅 2025/05/08

-   🔧 **開發**  
    新增嵌入與重排序伺服器：涵蓋模型加載、配置管理、API 路由處理

---

### 📅 2025/05/15

-   🛠️ **修復**  
    解決假流式回應的首字延遲問題，恢復正常表現

---

### 📅 2025/05/18

-   **優化（高併發）**
    -   共用 `AsyncClient`：降低連線開銷，併發 >10 吞吐提升 150%
    -   強化底層排程與 IO：併發 >100 回應時間縮短，首字延遲減少 30%
    -   啟用 gunicorn 多 worker：提升並行處理能力

---

**目前階段**：已完成伺服器基本架構，進入性能優化階段，具備支援多模型的可擴展性。

---

## A. 介紹
LLM Router Server 是一個針對多模型部署場景設計的輕量級路由服務，用於統一管理和調度多個本地 LLM（如 vLLM）、Embedding 模型、Re-ranking 模型等推理服務。它提供與 OpenAI 兼容的 API（如 `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`），方便接入現有應用與客戶端 SDK（如 `openai`）。
專案啟動支援配置式啟動，允許從 `config.yaml` 中自定義模型清單、每個模型的埠口、推理參數、以及 GPU 指派等資訊，實現**多模型獨立推理、集中轉發管理**。
## B. 特色
### 支援多個 LLM 實例獨立運行
- 每個模型透過獨立進程啟動，使用不同的埠與 CUDA
### 支援 Embedding 與 Reranker 伺服器整合
- Embedding/Reranker Server 與 Router 整合，轉發 `/v1/embeddings` 請求
- 同時支援向量嵌入與排序評分任務

### 完全相容 OpenAI SDK

### 流式請求穩定
- 經測試，Router Server 幾乎不影響請求延遲，保證首字延遲低、Token 穩定輸出

## C.用法
### 1. 設定模型配置
請參考範例 `configs/config.yaml`：
```yaml=
server:
  host: "0.0.0.0"
  port: 8947    
  uvicorn_log_level: "info"  
  loop: "uvloop"       

LLM_engines:
  Qwen2.5-1.5B-Instruct-GPTQ-Int4-2:
    model_tag: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
    host: "localhost"
    port: 8001
    dtype: "float16"
    max_model_len: 2000
    gpu_memory_utilization: 0.3
    max_num_seqs: 1
    quantization: "gptq"
    tensor_parallel_size: 1
  Qwen2.5-1.5B-Instruct-GPTQ-Int4:
    model_tag: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
    host: "localhost"
    port: 8002
    dtype: "float16"
    max_model_len: 2000
    gpu_memory_utilization: 0.3
    max_num_seqs: 1
    quantization: "gptq"
    tensor_parallel_size: 1
    cuda_device: 0
    enable_auto_tool_choice: true
    tool-call-parser: "hermes"
  Qwen2.5-1.5B-Instruct:
    model_tag: "Qwen/Qwen2.5-1.5B-Instruct"
    host: "localhost"
    port: 8003
    dtype: "float16"
    max_model_len: 32768
    gpu_memory_utilization: 0.95
    max_num_seqs: 200
    tensor_parallel_size: 1
    cuda_device: 0
    enable_auto_tool_choice: true
    tool-call-parser: "hermes"
    guided-decoding-backend: "xgrammar"

embedding_server:
  host: "localhost"
  port: 8004
  cuda_device: 0 

  embedding_models:
    m3e-base:
      model_name: "moka-ai/m3e-base"
      model_path: "./embedding_reranker_server/embedding_engine/model/embedding_model/m3e-base-model"
      tokenizer_path: "./embedding_reranker_server/embedding_engine/model/embedding_model/m3e-base-tokenizer"
      max_length: 512
      use_gpu: true
      use_float16: true

  reranking_models:
    bge-reranker-large:
      model_name: "BAAI/bge-reranker-large"
      model_path: "./embedding_reranker_server/embedding_engine/model/reranking_model/bge-reranker-large-model"
      tokenizer_path: "./embedding_reranker_server/embedding_engine/model/reranking_model/bge-reranker-large-tokenizer"
      max_length: 512
      use_gpu: true
      use_float16: true
```
### 2. 配置 `gunicorn.conf.py`
**1. 務必放在與 `main.py` 同個層級下**
**2. 路由配置務必與 `configs/config.yaml` 相同**
```python=
# gunicorn.conf.py

import os

bind = "0.0.0.0:8947"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 0
loglevel = "info"
accesslog = "-"  
errorlog = "-"   
preload_app = False  
```
### 啟動所有模型與路由服務
```bash=
sh start_all.sh ./configs/config.yaml
```
---

## D. 轉換ONNX 
基於 Optimum 框架，將 Transformer 模型轉換為 ONNX 格式，並進行優化以提升推理性能。支持特徵提取（Feature Extraction）模型和序列分類（Sequence Classification）模型的轉換與優化，適用於嵌入檢索和重排序任務。
**需額外安裝以下依賴**
- optimum[onnxruntime]
- transformers

### 嵌入模型的轉換
```python=
from optimize import optimize_embedding_model

# 設定參數
model_path = "./path_to_embedding_model"  # 輸入模型路徑
tokenizer_path = "./path_to_tokenizer"    # 輸入分詞器路徑
onnx_save_path = "./optimized_embedding_model"  # 優化後模型保存路徑

# 優化模型
optimize_embedding_model(
    model_path=model_path,
    tokenizer_path=tokenizer_path,
    onnx_save_path=onnx_save_path,
    optimization_level=2,    # 優化級別，數值越高優化效果越明顯，但可能影響穩定性。
    optimize_for_gpu=True,   # 是否針對 GPU
    fp16=True                # 是否啟用 FP16
)
```
### 重排序模型的轉換
```python=
from optimize import optimize_rerank_model

# 設定參數
model_path = "./path_to_rerank_model"  # 輸入模型路徑
tokenizer_path = "./path_to_tokenizer" # 輸入分詞器路徑
onnx_save_path = "./optimized_rerank_model"  # 優化後模型保存路徑

# 優化模型
optimize_rerank_model(
    model_path=model_path,
    tokenizer_path=tokenizer_path,
    onnx_save_path=onnx_save_path,
    optimization_level=2,    # 優化級別，數值越高優化效果越明顯，但可能影響穩定性。
    optimize_for_gpu=True,   # 是否針對 GPU
    fp16=True                # 是否啟用 FP16
)
```