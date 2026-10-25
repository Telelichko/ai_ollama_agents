import asyncio
import os
from pathlib import Path

from .agents import RESEARCHER, CODER, REVIEWER
from .ollama_client import OLLAMA_MODEL

DEFAULT_TASK = 'Briefly describe 3 approaches to caching in Python.'


def load_task() -> str:
    task = os.getenv('TASK')
    if task and task.strip():
        return task.strip()

    p = Path(os.getenv('TASK_FILE', 'data/task.txt'))
    if p.exists():
        text = p.read_text(encoding='utf-8').strip()
        if text:
            return text

    return DEFAULT_TASK


def save_result(text: str) -> None:
    out = Path(os.getenv('RESULT_FILE', 'data/result.md'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    print(f'\n💾 Результат сохранён: {out.resolve()}')


async def pipeline(task: str) -> str:
    print('=== PIPELINE ===')
    print(f'Модель: {OLLAMA_MODEL}')
    print(f'Задача: {task}\n')

    research = await RESEARCHER.run(task)
    print('\n--- RESEARCHER OUTPUT ---\n' + research)

    code = await CODER.run(f'Задача: {task}\n\nИсследование:\n{research}')
    print('\n--- CODER OUTPUT ---\n' + code)

    review = await REVIEWER.run(
        f'Задача: {task}\n\nИсследование:\n{research}\n\nРешение:\n{code}'
    )
    print('\n--- REVIEWER OUTPUT ---\n' + review)

    return (
        f'# Задача\n{task}\n\n'
        f'## Исследование\n{research}\n\n'
        f'## Решение\n{code}\n\n'
        f'## Ревью\n{review}\n'
    )


async def main() -> None:
    task = load_task()
    result = await pipeline(task)
    save_result(result)


if __name__ == '__main__':
    asyncio.run(main())
