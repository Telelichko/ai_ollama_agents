from dataclasses import dataclass
from .ollama_client import chat


@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str

    async def run(self, task: str) -> str:
        print(f'[{self.role}] 🚀 task: {task[:80]}...')
        return await chat(prompt=task, system=self.system_prompt)


RESEARCHER = Agent(
    name='researcher',
    role='Researcher',
    system_prompt=(
        'Ты — исследователь. Кратко и по делу разбираешь задачу, '
        'перечисляешь ключевые факты, подходы и подводные камни. '
        'Пиши структурированно, без воды.'
    ),
)

CODER = Agent(
    name='coder',
    role='Coder',
    system_prompt=(
        'Ты — инженер-программист. На основе исследования пишешь '
        'конкретное решение: код, команды, конфиги. Сопровождай код '
        'краткими пояснениями. Используй Python, если не сказано иное.'
    ),
)

REVIEWER = Agent(
    name='reviewer',
    role='Reviewer',
    system_prompt=(
        'Ты — ревьюер. Проверяешь решение на корректность, '
        'указываешь на ошибки, риски и как их исправить. '
        'В конце дай краткое резюме: годится / не годится.'
    ),
)
