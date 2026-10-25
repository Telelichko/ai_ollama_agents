import asyncio
from .agents import RESEARCHER, CODER, REVIEWER
from .tasks import PIPELINE

AGENTS = {'Researcher': RESEARCHER, 'Coder': CODER, 'Reviewer': REVIEWER}

async def pipeline():
    context = ''
    for agent_name, task in PIPELINE:
        agent = AGENTS[agent_name]
        full_task = f'{task}\n\nContext:\n{context}' if context else task
        context = await agent.run(full_task)
        print(f'\n--- {agent_name} Result ---\n{context}\n')

async def parallel_fan_out():
    """Example: run N agents simultaneously."""
    tasks = [
        RESEARCHER.run('What is RAG (Retrieval-Augmented Generation)?'),
        CODER.run('Write a SHA-256 hash function in Python.'),
        REVIEWER.run('List the pros and cons of microservices architecture.'),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print('\n=====\n', r)

async def main():
    print('=== PIPELINE ===')
    await pipeline()
    print('\n=== PARALLEL ===')
    await parallel_fan_out()

if __name__ == "__main__":
    asyncio.run(main())