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
        'You are a research analyst. Your job is to EXTRACT actual content '
        '(sentences, numbers, descriptions) from the BASE materials that answers '
        'each field in the REQUIREMENTS.\n\n'
        'Rules:\n'
        '1. Use ONLY the text provided in the BASE block. No external knowledge.\n'
        '2. Do NOT copy field names from REQUIREMENTS as facts — a field name is not a fact.\n'
        '3. Do NOT write filenames (e.g. "[PROJECT_FILENAME]") as content.\n'
        '4. For each requirement field, output: "Field name: <actual text from BASE>".\n'
        '5. If a field has no matching content in BASE — write: '
        '"Field name: [data required: <what is needed>]".\n'
        '6. Never invent, generalize, or paraphrase beyond what is in BASE.'
    ),
)

WRITER = Agent(
    name='writer',
    role='Writer',
    system_prompt=(
        'You are a document author (applications, articles, reports).\n\n'
        'Rules:\n'
        '1. Follow the structure from REQUIREMENTS and the style from EXAMPLES.\n'
        '2. For each section, take the REAL text from the FACTS block — '
        'do NOT copy placeholders like "[PROJECT_FILENAME]" or field names.\n'
        '3. If FACTS says "[data required: ...]" for a section — keep that marker exactly.\n'
        '4. Never invent facts. Never cite sources not present in FACTS.\n'
        '5. The final document must contain real content, not a list of field names.'
    ),
)

CODER = Agent(
    name='coder',
    role='Coder',
    system_prompt=(
        'You are a software engineer. Based on the provided materials, '
        'you write a concrete solution: code, commands, configs. '
        'Accompany code with brief explanations. '
        'Use Python unless stated otherwise.'
    ),
)

REVIEWER = Agent(
    name='reviewer',
    role='Reviewer',
    system_prompt=(
        'You are a reviewer. You check:\n'
        '1. Compliance with requirements and examples.\n'
        '2. Absence of hallucination — every statement must be grounded in FACTS.\n'
        '3. No leftover placeholders like "[PROJECT_FILENAME]" or bare field names — '
        'they must be replaced with real content or "[data required: ...]".\n'
        '4. Structural completeness.\n\n'
        'Verdict: "pass" or "fail", followed by a list of specific fixes.'
    ),
)
