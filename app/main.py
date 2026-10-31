import asyncio
import json
import os
from pathlib import Path

from .agents import RESEARCHER, WRITER, CODER, REVIEWER
from .ollama_client import OLLAMA_MODEL

CONFIG_PATH = Path(os.getenv('PIPELINE_CONFIG', 'config/pipeline.json'))
NOT_PROVIDED = '(not provided)'


def read_safe(p: Path) -> str:
    """Reads a file as UTF-8 text; returns empty string on decode/IO error."""
    try:
        return p.read_text(encoding='utf-8').strip()
    except (UnicodeDecodeError, OSError):
        return ''


def read_folder(dir_path: Path, label: str) -> str:
    """Reads all files from a folder, tagging each with its source label [LABEL]."""
    if not dir_path.is_dir():
        return ''
    chunks = []
    for f in sorted(dir_path.iterdir()):
        if not f.is_file():
            continue
        text = read_safe(f)
        if text:
            chunks.append(f'### [{label}] {f.name}\n{text}')
    return '\n\n'.join(chunks)


def read_task_file(base: Path) -> str:
    """Reads the task description from base/task.txt."""
    return read_safe(base / 'task.txt')


def build_research_context(task: dict) -> dict[str, str]:
    """Collects task, requirements, examples, and base materials for a research task."""
    base = Path(task['base_path'])
    folders = task.get('folders', {})
    return {
        'task': read_task_file(base),
        'requirements': read_folder(
            base / folders.get('requirements', 'requirements'), 'REQUIREMENTS'
        ),
        'examples': read_folder(
            base / folders.get('examples', 'examples'), 'EXAMPLES'
        ),
        'base': read_folder(
            base / folders.get('base', 'base'), 'BASE'
        ),
    }


def build_development_context(task: dict) -> str:
    """Collects task description and files for a development task."""
    base = Path(task['base_path'])
    chunks = []

    task_text = read_task_file(base)
    if task_text:
        chunks.append(f'### [TASK] task.txt\n{task_text}')

    files = read_folder(base, 'FILES')
    if files:
        chunks.append(files)

    if not chunks:
        raise ValueError(f'No files in {base.resolve()}')
    return '\n\n'.join(chunks)


async def run_research_pipeline(ctx: dict[str, str]) -> str:
    """Runs the research pipeline: extract facts -> compose document -> review."""
    task_text = ctx['task'] or '(task not specified in task.txt)'
    req = ctx['requirements'] or NOT_PROVIDED
    ex = ctx['examples'] or NOT_PROVIDED
    base = ctx['base'] or NOT_PROVIDED

    # Step 1: extract facts strictly from the provided materials
    facts = await RESEARCHER.run(
        f'TASK:\n{task_text}\n\n'
        f'REQUIREMENTS:\n{req}\n\n'
        f'EXAMPLES:\n{ex}\n\n'
        f'BASE:\n{base}\n\n'
        f'Extract from these materials only the facts, data, wording, '
        f'structures, and templates needed to complete the task. '
        f'Tag every fact with its source as [LABEL] filename. '
        f'If something is missing — write "not provided". Do not invent.'
    )
    print(f'\n--- RESEARCHER FACTS ---\n{facts}')

    # Step 2: compose the document strictly from requirements and facts
    document = await WRITER.run(
        f'TASK:\n{task_text}\n\n'
        f'REQUIREMENTS:\n{req}\n\n'
        f'EXAMPLES:\n{ex}\n\n'
        f'BASE / STYLE:\n{base}\n\n'
        f'FACTS (the only permitted source):\n{facts}\n\n'
        f'Compose the document per the task and requirements, following the '
        f'structure and style from the examples and base. Use only facts from '
        f'the FACTS block. Where data is missing — insert [data required: <what is needed>].'
    )
    print(f'\n--- WRITER DOCUMENT ---\n{document}')

    # Step 3: review for hallucination and compliance
    review = await REVIEWER.run(
        f'TASK:\n{task_text}\n\n'
        f'REQUIREMENTS:\n{req}\n\n'
        f'DOCUMENT:\n{document}\n\n'
        f'Check: (1) compliance with requirements and examples; '
        f'(2) absence of hallucination — every statement must be grounded '
        f'in the provided materials; (3) structural completeness. '
        f'Verdict: "pass" / "fail", then a list of fixes.'
    )
    print(f'\n--- REVIEWER ---\n{review}')

    return (
        f'# Research Report\n\n'
        f'## Task\n{task_text}\n\n'
        f'## Facts\n{facts}\n\n'
        f'## Document\n{document}\n\n'
        f'## Review\n{review}\n'
    )


async def run_development_pipeline(context: str) -> str:
    """Runs the development pipeline: analyze -> write code -> review."""
    analysis = await RESEARCHER.run(
        f'Task and materials:\n{context}\n\n'
        f'Extract requirements, constraints, and inputs. '
        f'Propose a solution architecture. Do not invent external facts.'
    )
    print(f'\n--- RESEARCHER ANALYSIS ---\n{analysis}')

    code = await CODER.run(
        f'Task and materials:\n{context}\n\nAnalysis:\n{analysis}\n\nWrite the code.'
    )
    print(f'\n--- CODER ---\n{code}')

    review = await REVIEWER.run(
        f'Task and materials:\n{context}\n\nCode:\n{code}\n\nReview and give a verdict.'
    )
    print(f'\n--- REVIEWER ---\n{review}')

    return (
        f'# Dev Report\n\n'
        f'## Analysis\n{analysis}\n\n'
        f'## Code\n{code}\n\n'
        f'## Review\n{review}\n'
    )


async def main():
    """Loads config, runs all tasks, and writes each result to its output file."""
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    model = config.get('settings', {}).get('model', OLLAMA_MODEL)
    temperature = config.get('settings', {}).get('temperature', 0.2)
    print(f'🚀 Starting | Model: {model} | temperature: {temperature}')

    for task in config['tasks']:
        name = task.get('name', 'unnamed')
        kind = task['task_type']
        print(f'\n📌 {name} ({kind})')

        try:
            if kind == 'research':
                ctx = build_research_context(task)
                result = await run_research_pipeline(ctx)
            elif kind == 'development':
                ctx = build_development_context(task)
                result = await run_development_pipeline(ctx)
            else:
                raise ValueError(f'Unknown task_type: {kind}')

            output = Path(task.get('output_file', f'data/results/{name}.md'))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result, encoding='utf-8')
            print(f'✅ Saved: {output}')
        except Exception as e:
            print(f'❌ Error: {e}')

    print('\n🎉 Done!')


if __name__ == '__main__':
    asyncio.run(main())
