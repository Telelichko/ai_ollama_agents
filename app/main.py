import asyncio
import json
import os
from pathlib import Path

from .agents import RESEARCHER, WRITER, CODER, REVIEWER
from .ollama_client import OLLAMA_MODEL

CONFIG_PATH = Path(os.getenv('PIPELINE_CONFIG', 'config/pipeline.json'))
NOT_PROVIDED = '(not provided)'


# ---------- file readers ----------

def read_text_safe(p: Path) -> str:
    """Reads a text file as UTF-8; returns empty string on error."""
    try:
        return p.read_text(encoding='utf-8').strip()
    except (UnicodeDecodeError, OSError):
        return ''


def read_pdf(p: Path) -> str:
    """Extracts text from a PDF using pypdf. Returns '' on failure."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        if reader.is_encrypted:
            return ''
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or '').strip()
            if text:
                pages.append(f'--- page {i} ---\n{text}')
        return '\n\n'.join(pages)
    except Exception:
        return ''


def read_file(p: Path) -> str:
    """Dispatches to the right reader based on file extension."""
    ext = p.suffix.lower()
    if ext == '.pdf':
        return read_pdf(p)
    if ext in {'.txt', '.md', '.rst', '.json', '.yaml', '.yml', '.csv', '.xml'}:
        return read_text_safe(p)
    return ''


def read_folder(dir_path: Path, label: str) -> str:
    """Reads all files from a folder, tagging each with [LABEL] filename."""
    if not dir_path.is_dir():
        return ''
    chunks = []
    for f in sorted(dir_path.iterdir()):
        if not f.is_file():
            continue
        text = read_file(f)
        if text:
            chunks.append(f'### [{label}] {f.name}\n{text}')
    return '\n\n'.join(chunks)


def read_task_file(base: Path) -> str:
    return read_text_safe(base / 'task.txt')


# ---------- context builders ----------

def build_research_context(task: dict) -> dict[str, str]:
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


# ---------- pipelines ----------

async def run_research_pipeline(ctx: dict[str, str]) -> str:
    task_text = ctx['task'] or '(task not specified in task.txt)'
    req = ctx['requirements'] or NOT_PROVIDED
    ex = ctx['examples'] or NOT_PROVIDED
    base = ctx['base'] or NOT_PROVIDED

    # Step 1: extract facts strictly from BASE materials
    facts = await RESEARCHER.run(
        f'TASK:\n{task_text}\n\n'
        f'REQUIREMENTS (structure only — do NOT copy field names as facts):\n{req}\n\n'
        f'EXAMPLES:\n{ex}\n\n'
        f'BASE (the ONLY source of facts):\n{base}\n\n'
        f'Extract from BASE only the actual content (sentences, numbers, descriptions) '
        f'that answers each requirement field. '
        f'For each requirement field, output one bullet: "Field name: <actual content from BASE>". '
        f'If a field has no matching content in BASE — write "Field name: [data required: <what is needed>]". '
        f'Do NOT copy field names from REQUIREMENTS as facts. '
        f'Do NOT invent anything. Do NOT write filenames as content.'
    )
    print(f'\n--- RESEARCHER FACTS ---\n{facts}')

    # Step 2: compose the document strictly from facts
    document = await WRITER.run(
        f'TASK:\n{task_text}\n\n'
        f'REQUIREMENTS (follow this structure):\n{req}\n\n'
        f'EXAMPLES (follow this style):\n{ex}\n\n'
        f'FACTS (the ONLY permitted source of content):\n{facts}\n\n'
        f'Compose the document per the task and requirements. '
        f'For each section, take the real text from FACTS — do NOT copy placeholders '
        f'like "[PROJECT_FILENAME]" or field names. '
        f'If FACTS says "[data required: ...]" for a section — keep that marker exactly. '
        f'Never invent facts. Never cite sources not present in FACTS.'
    )
    print(f'\n--- WRITER DOCUMENT ---\n{document}')

    # Step 3: review
    review = await REVIEWER.run(
        f'TASK:\n{task_text}\n\n'
        f'REQUIREMENTS:\n{req}\n\n'
        f'FACTS:\n{facts}\n\n'
        f'DOCUMENT:\n{document}\n\n'
        f'Check: (1) compliance with requirements and examples; '
        f'(2) absence of hallucination — every statement must be grounded in FACTS; '
        f'(3) no placeholders like "[PROJECT_FILENAME]" left in the document — '
        f'they must be replaced with real content or "[data required: ...]"; '
        f'(4) structural completeness. '
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
