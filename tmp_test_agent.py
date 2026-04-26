# tmp_test_agent.py
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
    # Case 1: query faktual sederhana
    await test("What is attention mechanism in transformers?")

    # Case 2: query reasoning
    await test("Why does self-attention work better than RNN for long sequences?")

    # Case 3: query out of domain — harusnya ABSTAINED atau confidence rendah
    await test("What is the recipe for nasi goreng?")

asyncio.run(main())