import asyncio
import json
import os
from pathlib import Path

from .agents import RESEARCHER, CODER, REVIEWER
from .ollama_client import OLLAMA_MODEL

CONFIG_PATH = Path(os.getenv('PIPELINE_CONFIG', 'config/pipeline.json'))


def read_files(dir_path: Path) -> str:
    """Reads all text files from directory."""
    if not dir_path.exists():
        return ''

    texts = []
    for f in sorted(dir_path.iterdir()):
        if f.is_file():
            try:
                texts.append(f'--- {f.name} ---\n{f.read_text(encoding="utf-8")}\n')
            except UnicodeDecodeError:
                pass
    return '\n'.join(texts)


def build_context(task: dict) -> str:
    """Builds context from task config."""
    base = Path(task['base_path'])

    if task['task_type'] == 'research':
        folders = task.get('folders', {})
        return (
            '=== RESEARCH CONTEXT ===\n\n'
            f'## Requirements\n{read_files(base / folders.get("requirements", "requirements"))}\n\n'
            f'## Examples\n{read_files(base / folders.get("examples", "examples"))}\n\n'
            f'## Base Materials\n{read_files(base / folders.get("base", "base"))}'
        ).strip()

    elif task['task_type'] == 'development':
        text = read_files(base)
        if not text:
            raise ValueError(f'No files in {base}')
        return f'=== DEVELOPMENT TASK ===\n\n{text}'

    raise ValueError(f'Unknown type: {task["task_type"]}')


async def run_pipeline(task_type: str, context: str) -> str:
    """Runs appropriate pipeline based on task type."""
    if task_type == 'development':
        research = await RESEARCHER.run(f'Analyze and propose architecture:\n\n{context}')
        code = await CODER.run(f'Task:\n{context}\n\nApproach:\n{research}\n\nWrite code.')
        review = await REVIEWER.run(f'Task:\n{context}\n\nCode:\n{code}\n\nReview.')
        return f'# Dev Report\n\n## Analysis\n{research}\n\n## Code\n{code}\n\n## Review\n{review}\n'

    else:  # research
        draft = await RESEARCHER.run(f'Write material following requirements:\n\n{context}')
        review = await REVIEWER.run(f'Check material quality:\n\n{draft}')
        return f'# Research Report\n\n## Draft\n{draft}\n\n## Review\n{review}\n'


async def main():
    config = json.loads(Path(CONFIG_PATH).read_text(encoding='utf-8'))
    print(f'🚀 Starting | Model: {config.get("settings", {}).get("model", OLLAMA_MODEL)}')

    for task in config['tasks']:
        name = task.get('name', 'unnamed')
        print(f'\n📌 {name} ({task["task_type"]})')

        try:
            context = build_context(task)
            result = await run_pipeline(task['task_type'], context)

            output = Path(task.get('output_file', f'data/results/{name}.md'))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result, encoding='utf-8')
            print(f'✅ Saved: {output}')
        except Exception as e:
            print(f'❌ Error: {e}')

    print('\n🎉 Done!')


if __name__ == '__main__':
    asyncio.run(main())
