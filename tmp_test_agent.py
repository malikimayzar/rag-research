import asyncio
import logging
from src.controller.agent import Agent

logging.basicConfig(level=logging.INFO, format="%(message)s")

agent = Agent()

async def test(query: str):
    print(f"\n[QUERY] {query}")
    response = await agent.run(query)
    print(f"  status : {response.status}")
    print(f"  answer : {response.answer[:200] if response.answer else None}")

async def main():
    await test("What is attention mechanism in transformers?")
    await test("Why does self-attention work better than RNN for long sequences?")
    await test("What is the recipe for nasi goreng?")
asyncio.run(main())