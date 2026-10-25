import httpx
from pydantic import BaseModel

OLLAMA_URL = "http://ollama:11434"

class Message(BaseModel):
    role: str
    content: str

async def chat(model: str, messages: list[Message], temperature: float = 0.7) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f'{OLLAMA_URL}/api/chat',
            json={
                'model': model,
                'messages': [m.model_dump() for m in messages],
                'stream': False,
                'options': {'temperature': temperature},
            },
        )
        r.raise_for_status()
        return r.json()['message']['content']