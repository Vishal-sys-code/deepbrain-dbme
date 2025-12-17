import json
import random
import os

def generate_sessions(num_sessions=1000, output_file="data/retention_sessions.json"):
    """
    Generates synthetic sessions with injected facts.
    """
    sessions = []
    
    cities = ["Zurich", "London", "Paris", "Berlin", "New York", "Tokyo", "San Francisco", "Toronto", "Sydney", "Mumbai"]
    
    fact_registry = []

    print(f"Generating {num_sessions} sessions...")

    for i in range(num_sessions):
        session_id = i
        session_length = random.randint(5, 15) # Random session length
        
        session_data = {
            "session_id": session_id,
            "turns": [],
            "injected_facts": []
        }

        # Inject a fact with some probability or fixed schedule. 
        # For this benchmark, let's inject 1 fact per session to specific entities.
        # Actually prompt says: "Unique entities, Non-overlapping facts"
        
        person_id = f"Person_{i}" 
        city = random.choice(cities)
        fact_text = f"{person_id} lives in {city}."
        
        # Random insertion time within the session
        insertion_index = random.randint(0, session_length - 1)
        
        for t in range(session_length):
            if t == insertion_index:
                # Insert fact
                turn_text = f"User: Tell me about {person_id}.\nSystem: {fact_text}"
                session_data["turns"].append(turn_text)
                
                # Record the injected fact
                fact_entry = {
                    "session_id": session_id,
                    "time": t, # relative time in session, or we can track global time
                    "fact_id": f"f_{i}",
                    "fact_text": fact_text,
                    "entity": person_id,
                    "attribute": "lives in",
                    "value": city
                }
                session_data["injected_facts"].append(fact_entry)
                fact_registry.append(fact_entry)
            else:
                # Filler text
                turn_text = f"User: Chat {t}\nSystem: Response {t}"
                session_data["turns"].append(turn_text)
        
        sessions.append(session_data)

    # Save sessions
    with open(output_file, "w") as f:
        json.dump(sessions, f, indent=2)
    
    # Save the ground truth facts separately for easy query generation
    with open("data/ground_truth_facts.json", "w") as f:
        json.dump(fact_registry, f, indent=2)

    print(f"Saved {len(sessions)} sessions to {output_file}")
    print(f"Saved {len(fact_registry)} facts to data/ground_truth_facts.json")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    generate_sessions()
