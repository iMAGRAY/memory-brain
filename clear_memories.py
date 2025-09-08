#!/usr/bin/env python3
"""
Clear all memories from the database to reimport with fixed embeddings
"""
import asyncio
import aiohttp
import json

async def clear_all_memories():
    """Clear all memories from the database"""
    print("🗑️ Clearing all memories from database...")
    
    async with aiohttp.ClientSession() as session:
        # First, get count of existing memories
        async with session.get("http://localhost:8080/memories") as response:
            if response.status == 200:
                memories = await response.json()
                count = len(memories.get("memories", []))
                print(f"📊 Found {count} memories to delete")
            else:
                print(f"❌ Failed to get memories: {response.status}")
                return False
        
        # Delete all memories (assuming there's an endpoint for this)
        # Since I don't see a bulk delete endpoint, we might need to delete them one by one
        # or clear the database manually
        
        print("⚠️ Note: This script would need to implement actual deletion")
        print("   For now, you may need to manually clear Neo4j database:")
        print("   MATCH (n) DETACH DELETE n")
        
        return True

if __name__ == "__main__":
    asyncio.run(clear_all_memories())