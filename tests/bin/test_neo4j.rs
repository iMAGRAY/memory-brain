// Neo4j connection test without PyO3 dependencies
use ai_memory_service::storage::GraphStorage;
use ai_memory_service::types::{MemoryCell, MemoryType};
use uuid::Uuid;
use tokio;

#[tokio::main]
async fn main() {
    println!("Testing Neo4j connection and operations...\n");
    
    // Neo4j connection parameters
    let neo4j_uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string());
    let neo4j_user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string());
    let neo4j_pass = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string());
    
    println!("Connecting to Neo4j at: {}", neo4j_uri);
    println!("Username: {}", neo4j_user);
    
    // Try to create storage
    let storage = match GraphStorage::new(&neo4j_uri, &neo4j_user, &neo4j_pass).await {
        Ok(s) => {
            println!("✅ Successfully connected to Neo4j!\n");
            s
        }
        Err(e) => {
            println!("❌ Failed to connect to Neo4j: {}", e);
            println!("\nMake sure Neo4j is running and accessible at {}.", neo4j_uri);
            println!("You can start Neo4j with: docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:latest");
            return;
        }
    };
    
    println!("Testing Neo4j operations:");
    
    // Test 1: Store a memory
    println!("\n1. Testing memory storage...");
    let memory = MemoryCell {
        id: Uuid::new_v4(),
        content: "Test memory content for Neo4j integration".to_string(),
        summary: "Test memory for Neo4j".to_string(),
        tags: vec!["test".to_string(), "neo4j".to_string()],
        embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        memory_type: MemoryType::Semantic {
            facts: vec!["Neo4j test".to_string()],
            concepts: vec!["testing".to_string(), "integration".to_string()],
        },
        importance: 0.8,
        access_frequency: 0,
        created_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
        context_path: "/test/neo4j".to_string(),
        metadata: std::collections::HashMap::new(),
    };
    
    match storage.store_memory(&memory).await {
        Ok(_) => println!("  ✓ Memory stored successfully"),
        Err(e) => {
            println!("  ✗ Failed to store memory: {}", e);
            return;
        }
    }
    
    // Test 2: Retrieve the memory
    println!("\n2. Testing memory retrieval...");
    match storage.get_memory(&memory.id).await {
        Ok(retrieved) => {
            println!("  ✓ Memory retrieved successfully");
            println!("    ID: {}", retrieved.id);
            println!("    Content: {}", retrieved.content);
            println!("    Type: {:?}", retrieved.memory_type);
        }
        Err(e) => {
            println!("  ✗ Failed to retrieve memory: {}", e);
            return;
        }
    }
    
    // Test 3: Search memories by context
    println!("\n3. Testing context search...");
    match storage.get_memories_in_context("/test", 10).await {
        Ok(memories) => {
            println!("  ✓ Found {} memories in context '/test'", memories.len());
            for (i, mem) in memories.iter().take(3).enumerate() {
                println!("    {}. {}", i + 1, mem.content.chars().take(50).collect::<String>());
            }
        }
        Err(e) => println!("  ✗ Failed to search by context: {}", e),
    }
    
    // Test 4: Update memory stats
    println!("\n4. Testing memory stats update...");
    match storage.get_memory_stats().await {
        Ok(stats) => {
            println!("  ✓ Memory stats retrieved:");
            println!("    Total memories: {}", stats.total_memories);
            println!("    Total contexts: {}", stats.total_contexts);
        }
        Err(e) => println!("  ✗ Failed to get stats: {}", e),
    }
    
    // Test 5: Delete the test memory
    println!("\n5. Testing memory deletion...");
    match storage.delete_memory(&memory.id).await {
        Ok(_) => println!("  ✓ Memory deleted successfully"),
        Err(e) => println!("  ✗ Failed to delete memory: {}", e),
    }
    
    // Test 6: Verify deletion
    println!("\n6. Verifying deletion...");
    match storage.get_memory(&memory.id).await {
        Err(e) if e.to_string().contains("not found") => {
            println!("  ✓ Memory successfully deleted (not found)")
        },
        Ok(_) => println!("  ✗ Memory still exists after deletion"),
        Err(e) => println!("  ✗ Error checking deletion: {}", e),
    }
    
    println!("\n✅ Neo4j integration tests completed!");
}