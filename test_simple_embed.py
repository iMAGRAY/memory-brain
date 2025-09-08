#!/usr/bin/env python3
"""
Simple test to check embedding dimensions and task_type
"""
import asyncio
import aiohttp
import json

async def test_embeddings():
    test_text = "Claude Code hooks for development"
    
    async with aiohttp.ClientSession() as session:
        # Test query
        async with session.post(
            "http://localhost:8090/embed",
            json={"text": test_text, "task_type": "query"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Query embedding: {len(result['embedding'])} dimensions")
            else:
                print(f"❌ Query failed: {response.status}")
        
        # Test document  
        async with session.post(
            "http://localhost:8090/embed",
            json={"text": test_text, "task_type": "document"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Document embedding: {len(result['embedding'])} dimensions")
            else:
                print(f"❌ Document failed: {response.status}")
        
        # Test general
        async with session.post(
            "http://localhost:8090/embed",
            json={"text": test_text, "task_type": "general"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ General embedding: {len(result['embedding'])} dimensions")
            else:
                print(f"❌ General failed: {response.status}")

if __name__ == "__main__":
    asyncio.run(test_embeddings())