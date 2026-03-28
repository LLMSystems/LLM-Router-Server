import asyncio
import re
from dataclasses import dataclass
from typing import Dict, Optional

import httpx


@dataclass
class VLLMInstanceMetrics:
    base_url: str
    running: float = 0.0
    waiting: float = 0.0
    kv_cache_usage_perc: float = 0.0
    prompt_tokens: float = 0.0
    generation_tokens: float = 0.0
    raw_metrics: Optional[str] = None
    
    def compute_load_score(
        self,
        waiting_weight: float = 10.0,
        running_weight: float = 3.0,
        kv_cache_weight: float = 100.0,
    ) -> float:
        """
        A simple heuristic score for load-aware routing.
        Lower score means the backend is less loaded.
        """
        return (
            self.waiting * waiting_weight
            + self.running * running_weight
            + self.kv_cache_usage_perc * kv_cache_weight
        )
        
        
class VLLMMetricsClient:
    METRIC_NAMES = {
        "running": "vllm:num_requests_running",
        "waiting": "vllm:num_requests_waiting",
        "kv_cache_usage_perc": "vllm:kv_cache_usage_perc",
        "prompt_tokens": "vllm:prompt_tokens",
        "generation_tokens": "vllm:generation_tokens",
    }
    
    def __init__(self, timeout: float = 2.0) -> None:
        self.timeout = timeout
        
    async def fetch(self, base_url: str) -> Optional[VLLMInstanceMetrics]:
        metrics_url = base_url.rstrip("/") + "/metrics"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(metrics_url)
            resp.raise_for_status()
            text = resp.text
        
        parsed = self.parse_metrics(text)
            
        return VLLMInstanceMetrics(
            base_url=base_url,
            running=parsed.get(self.METRIC_NAMES["running"], 0.0),
            waiting=parsed.get(self.METRIC_NAMES["waiting"], 0.0),
            kv_cache_usage_perc=parsed.get(
                self.METRIC_NAMES["kv_cache_usage_perc"], 0.0
            ),
            prompt_tokens=parsed.get(self.METRIC_NAMES["prompt_tokens"], 0.0),
            generation_tokens=parsed.get(
                self.METRIC_NAMES["generation_tokens"], 0.0
            ),
            raw_metrics=text,
        )
        
    async def _safe_fetch(
        self,
        backend_name: str,
        base_url: str,
    ) -> tuple[str, VLLMInstanceMetrics]:
        try:
            metrics = await self.fetch(base_url)
            return backend_name, metrics
        except Exception:
            return backend_name, VLLMInstanceMetrics(
                base_url=base_url,
                running=float("inf"),
                waiting=float("inf"),
                kv_cache_usage_perc=1.0,
                prompt_tokens=float("inf"),
                generation_tokens=float("inf"),
                raw_metrics=None,
            )
    
    async def fetch_many(
        self,
        backends: Dict[str, str],
    ) -> Dict[str, VLLMInstanceMetrics]:
        """
        Fetch metrics for many backends.

        Args:
            backends: mapping like
                {
                    "qwen14b-a": "http://127.0.0.1:8001",
                    "qwen14b-b": "http://127.0.0.1:8002",
                }

        Returns:
            Dict[str, VLLMInstanceMetrics]
        """       
         
        tasks = [
            self._safe_fetch(backend_name, base_url) for backend_name, base_url in backends.items()
        ]
        
        pairs = await asyncio.gather(*tasks)
        return dict(pairs)

    def parse_metrics(self, text: str) -> Dict[str, float]:
        """
        Parse Prometheus text exposition format into a metric-name -> value mapping.

        Notes:
        - Ignores # HELP / # TYPE lines
        - Supports both:
            vllm:num_requests_running 2
            vllm:num_requests_running{model_name="..."} 2
        - If the same metric appears multiple times with different labels,
          values are summed.
        """
        values: Dict[str, float] = {}

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parsed = self._parse_metric_line(line)
            if parsed is None:
                continue

            metric_name, metric_value = parsed
            values[metric_name] = values.get(metric_name, 0.0) + metric_value

        return values

    def _parse_metric_line(self, line: str) -> Optional[tuple[str, float]]:
        """
        Parse one Prometheus metric line.

        Examples:
            vllm:num_requests_running 2
            vllm:num_requests_running{model_name="Qwen"} 2
        """
        pattern = r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$'
        match = re.match(pattern, line)
        if not match:
            return None

        metric_name = match.group(1)
        metric_value = float(match.group(3))
        return metric_name, metric_value
    
