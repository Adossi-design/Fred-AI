"""Google Gemini provider (via Gemini's OpenAI-compatible API).

Uses the OpenAI SDK pointed at Google's OpenAI-compatible endpoint, so it
reuses the same chat / tool-calling / vision plumbing as the other providers.

Get a free API key at https://aistudio.google.com (then set GEMINI_API_KEY).
"""

import os
import json
from typing import Generator, List, Callable
from .base import BaseProvider, Message


class GeminiProvider(BaseProvider):
    """Google Gemini provider using the OpenAI-compatible endpoint."""

    name = "gemini"
    supports_streaming = True
    supports_vision = True
    supports_tools = True

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Task -> model mapping. Flash is fast, smart, multimodal, and free-tier
    # friendly; Pro is used for deep reasoning.
    TASK_MODELS = {
        "default": "gemini-2.5-flash",
        "balanced": "gemini-2.5-flash",
        "fast": "gemini-2.0-flash",
        "deep": "gemini-2.5-pro",
        "reasoning": "gemini-2.5-pro",
        "vision": "gemini-2.5-flash",
        "code": "gemini-2.5-flash",
        "chat": "gemini-2.0-flash",
    }
    MODELS = TASK_MODELS

    KNOWN_MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]
    AVAILABLE_MODELS = KNOWN_MODELS

    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)

        # Load API key: param > credentials.json > env
        if not api_key:
            try:
                from jarvis.auth.credentials import get_credential
                api_key = get_credential("gemini", "api_key")
            except ImportError:
                pass
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.base_url = kwargs.get("base_url") or os.getenv("GEMINI_BASE_URL", self.BASE_URL)

        # Allow model overrides from config
        config = kwargs.get("config", {})
        provider_cfg = config.get("providers", {}).get("gemini", {})
        models_cfg = provider_cfg.get("models", {}) or provider_cfg.get("task_models", {})
        if models_cfg:
            self.TASK_MODELS.update(models_cfg)
            self.MODELS = self.TASK_MODELS

        if not self.model:
            self.model = self.TASK_MODELS.get("default", "gemini-2.5-flash")

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        else:
            self.client = None

    def _convert_tools_to_openai(self, tools: List[Callable]) -> List[dict]:
        return self.convert_tools_to_schema(tools)

    def chat(
        self,
        messages: List[Message],
        system: str = None,
        stream: bool = True,
        **kwargs,
    ) -> Generator[str, None, None] | str:
        if not self.client:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY.")
        self.reset_stop()

        msg_list = []
        if system:
            msg_list.append({"role": "system", "content": system})
        for m in messages:
            if isinstance(m, dict):
                msg_list.append(m)
            else:
                msg_list.append({"role": m.role, "content": m.content})

        if stream:
            return self._chat_streaming(msg_list)
        return self._chat_non_streaming(msg_list)

    def _chat_streaming(self, msg_list: List[dict]) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self.model, messages=msg_list, stream=True
        )
        for chunk in response:
            if self._stop_flag:
                break
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _chat_non_streaming(self, msg_list: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=msg_list, stream=False
        )
        return response.choices[0].message.content

    def chat_with_tools(
        self,
        messages: List[dict],
        system: str = None,
        tools: List[Callable] = None,
    ):
        """Non-streaming chat that returns tool calls (used by the agent loop)."""
        if not self.client:
            raise ValueError("Gemini API key not configured")

        openai_tools = self._convert_tools_to_openai(tools) if tools else None

        msg_list = []
        if system:
            msg_list.append({"role": "system", "content": system})
        for m in messages:
            if isinstance(m, dict):
                msg_list.append(m)
            else:
                msg_list.append({"role": m.role, "content": m.content})

        kwargs = {"model": self.model, "messages": msg_list}
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        result = {"message": {"content": msg.content or "", "tool_calls": []}}
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                result["message"]["tool_calls"].append({
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": args},
                })
        if not result["message"]["tool_calls"]:
            result["message"]["tool_calls"] = None

        return type("Response", (), result)()

    def vision(self, image_path: str, prompt: str) -> str:
        """Analyze an image with Gemini (natively multimodal)."""
        if not self.client:
            raise ValueError("Gemini API key not configured")
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        vision_model = self.MODELS.get("vision", self.model)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ],
        }]
        response = self.client.chat.completions.create(
            model=vision_model, messages=messages, max_tokens=1024
        )
        return response.choices[0].message.content

    def list_models(self) -> List[str]:
        return self.KNOWN_MODELS

    async def discover_models(self) -> List:
        from .base import ModelInfo
        if self._discovered_models is not None:
            return self._discovered_models
        self._discovered_models = [ModelInfo(id=m, name=m) for m in self.KNOWN_MODELS]
        return self._discovered_models

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_default_model(self) -> str:
        return self.MODELS["default"]

    def get_model_for_task(self, task: str) -> str:
        return self.MODELS.get(task, self.MODELS["default"])

    def get_context_length(self, model: str = None) -> int:
        # Gemini 2.x flash/pro support ~1M token context.
        return 1_048_576

    def get_config_help(self) -> str:
        return (
            "Google Gemini\n\n"
            "1. Get a FREE API key: https://aistudio.google.com (Get API key)\n"
            "2. Set it:\n"
            "   export GEMINI_API_KEY=your-key   (or add GEMINI_API_KEY=... to ~/.jarvis/.env)\n\n"
            "Models: gemini-2.5-flash (default), gemini-2.5-pro (deep), gemini-2.0-flash (fast)."
        )
