from dataclasses import dataclass
from .ollama_client import chat, Message

@dataclass
class Agent:
    name: str
    model: str
    system_prompt: str

    async def run(self, task: str) -> str:
        print(f'[{self.name}] 🚀 task: {task[:80]}...')
        result = await chat(
            self.model,
            [Message(role='system', content=self.system_prompt),
             Message(role='user', content=task)],
        )
        print(f'[{self.name}] ✅ done')
        return result

# Specialized agents
RESEARCHER = Agent(
    name='Researcher',
    model='llama3.2:3b',
    system_prompt='You are a researcher. Provide facts, sources, and structure your answer clearly.',
)

CODER = Agent(
    name='Coder',
    model='qwen2.5-coder:3b',
    system_prompt='You are a Python developer. Write clean, efficient code with helpful comments.',
)

REVIEWER = Agent(
    name='Reviewer',
    model='mistral:7b',
    system_prompt='You are a code reviewer. Point out bugs, suggest improvements, and rate the quality from 1 to 10.',
)