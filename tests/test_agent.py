import asyncio
from src.controller.agent import Agent

async def main():
    agent = Agent()

    query = "what is attention mechanism in transformers"

    res = await agent.run(query)

    print("\n=== RESULT ===")
    print("Status:", res.status)
    print("Answer:", res.answer)

    print("\n=== DEBUG ===")
    print("Steps:", res.state.step_count)
    print("Confidence:", res.state.confidence_score)
    print("Chunks:", len(res.state.retrieved_chunks))

    if res.state.retrieved_chunks:
        print("\nTop chunk:")
        print(res.state.retrieved_chunks[0])

asyncio.run(main())