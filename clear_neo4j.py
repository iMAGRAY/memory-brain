#!/usr/bin/env python3
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import sys
import os

def clear_database():
    # Get Neo4j connection details from environment variables
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')  # Default for development
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    try:
        with driver.session() as session:
            # First check how many nodes exist
            result = session.run('MATCH (n) RETURN count(n) as node_count')
            node_count = result.single()['node_count']
            print(f'Found {node_count} nodes in database')
            
            if node_count > 0:
                print('Clearing database...')
                # Clear all nodes and relationships
                session.run('MATCH (n) DETACH DELETE n')
                print('Database cleared successfully')
                
                # Verify deletion
                result = session.run('MATCH (n) RETURN count(n) as remaining_count')
                remaining_count = result.single()['remaining_count']
                print(f'Remaining nodes after deletion: {remaining_count}')
            else:
                print('Database is already empty')
                
    except ServiceUnavailable as e:
        print(f'Neo4j service unavailable: {e}')
        print('Make sure Neo4j is running on the specified URI')
        return False
    except AuthError as e:
        print(f'Authentication error: {e}')
        print('Check your Neo4j credentials')
        return False
    except Exception as e:
        print(f'Unexpected error: {e}')
        return False
    finally:
        driver.close()
    
    return True

if __name__ == "__main__":
    success = clear_database()
    if not success:
        sys.exit(1)