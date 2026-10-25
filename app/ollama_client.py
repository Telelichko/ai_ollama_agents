import os
import httpx

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:3b')
OLLAMA_TIMEOUT = float(os.getenv('OLLAMA_TIMEOUT', '300'))

_client = httpx.AsyncClient(
    base_url=OLLAMA_URL,
    timeout=httpx.Timeout(OLLAMA_TIMEOUT),
)


async def chat(prompt: str, system: str | None = None, model: str | None = None) -> str:
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})

    payload = {
        'model': model or OLLAMA_MODEL,
        'messages': messages,
        'stream': False,
    }

    r = await _client.post('/api/chat', json=payload)
    r.raise_for_status()
    return r.json()['message']['content']
