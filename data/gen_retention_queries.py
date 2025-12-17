import json
import os

def generate_queries(facts_file="data/ground_truth_facts.json", output_file="data/retention_queries.json"):
    """
    Generates queries for facts at specific delays.
    """
    try:
        with open(facts_file, "r") as f:
            facts = json.load(f)
    except FileNotFoundError:
        print(f"Error: {facts_file} not found. Run gen_retention_sessions.py first.")
        return

    delays = [1, 10, 50, 100]
    queries = []

    print(f"Generating queries for {len(facts)} facts with delays {delays}...")

    for fact in facts:
        fact_id = fact["fact_id"]
        entity = fact["entity"]
        value = fact["value"]
        start_session_id = fact["session_id"]
        
        # Template for the query
        query_text = f"Where does {entity} live?"
        
        for delay in delays:
            trigger_session_id = start_session_id + delay
            
            # We assume total sessions = 1000. If trigger > 1000, we can still generate it, 
            # but it might not be reachable if we stop at 1000. 
            # Let's keep it, eval logic can handle bounds.
            
            query = {
                "query_id": f"q_{fact_id}_d{delay}",
                "fact_id": fact_id,
                "trigger_session_id": trigger_session_id,
                "query_text": query_text,
                "expected_answer": value,
                "delay": delay
            }
            queries.append(query)

    with open(output_file, "w") as f:
        json.dump(queries, f, indent=2)

    print(f"Saved {len(queries)} queries to {output_file}")

if __name__ == "__main__":
    generate_queries()
