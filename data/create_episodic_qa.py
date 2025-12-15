import json
import random
import argparse
from datetime import datetime, timedelta

# Simple knowledge base
FACTS = {
    "Paris": "France",
    "Berlin": "Germany",
    "Rome": "Italy",
    "Madrid": "Spain",
    "London": "the United Kingdom",
    "Tokyo": "Japan",
    "Beijing": "China",
    "Moscow": "Russia",
    "Cairo": "Egypt",
    "Washington, D.C.": "the United States",
}

def generate_session(session_id: int, facts_db: dict) -> dict:
    """Generates a single session with fact injections and delayed queries."""
    # Use a fixed date for deterministic output
    session_start_time = datetime(2023, 1, 1) - timedelta(days=random.randint(0, 30))
    session = {
        "session_id": f"s{session_id}",
        "events": []
    }

    # Select a subset of facts for this session
    num_facts = random.randint(2, 5)
    session_facts = random.sample(list(facts_db.items()), num_facts)

    # Inject facts at random times
    current_time = 0
    for city, country in session_facts:
        current_time += random.randint(1, 10) # Advance time
        session["events"].append({
            "t": current_time,
            "text": f"{city} is the capital of {country}."
        })

    # Add distractor statements
    num_distractors = random.randint(5, 15)
    distractors = [
        "The weather is nice today.",
        "I'm planning a trip soon.",
        "Remember to buy groceries.",
        "The stock market is volatile.",
        "Let's discuss the project timeline.",
        "My favorite movie is a classic.",
        "I learned a new recipe yesterday."
    ]
    for _ in range(num_distractors):
        current_time += random.randint(1, 20)
        session["events"].append({
            "t": current_time,
            "text": random.choice(distractors)
        })

    # Add delayed queries for the injected facts
    for city, country in session_facts:
        current_time += random.randint(10, 50) # Significant delay
        session["events"].append({
            "t": current_time,
            "query": f"What is the capital of {country}?",
            "answer": city
        })

    # Sort events by timestamp
    session["events"].sort(key=lambda x: x["t"])

    return session

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic episodic QA dataset.")
    parser.add_argument("--output_file", type=str, default="data/episodic_qa.jsonl",
                        help="Path to the output JSONL file.")
    parser.add_argument("--num_sessions", type=int, default=1000,
                        help="Number of sessions to generate.")
    args = parser.parse_args()

    with open(args.output_file, 'w') as f:
        for i in range(args.num_sessions):
            session = generate_session(i, FACTS)
            f.write(json.dumps(session) + '\n')

    print(f"Successfully generated {args.num_sessions} sessions to {args.output_file}")

if __name__ == "__main__":
    main()