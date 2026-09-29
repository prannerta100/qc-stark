import time
import os
from openai import OpenAI
from qc_hard.models.base import BaseModel
from qc_hard.types import TaskInstance, ModelResponse

from qc_hard.prompts import SYSTEM_PROMPT

# Endpoint is supplied by the environment; no default is baked in.
DEFAULT_BASE_URL = os.environ.get("QCSTARK_LLM_BASE_URL", "")


class ProxyModel(BaseModel):
    def __init__(self, name: str, model_id: str, base_url: str | None = None, api_key: str | None = None):
        self.name = name
        self._model_id = model_id
        self._client = OpenAI(
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=api_key or os.environ.get("QCSTARK_LLM_API_KEY", ""),
        )

    def query(self, task: TaskInstance) -> ModelResponse:
        start = time.time()
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        latency = (time.time() - start) * 1000
        content = response.choices[0].message.content or ""
        return ModelResponse(
            task_id=task.task_id,
            model_name=self.name,
            raw_response=content,
            parsed_code=self._extract_code(content),
            latency_ms=latency,
            token_count=response.usage.total_tokens if response.usage else 0,
        )

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        return text.strip()
