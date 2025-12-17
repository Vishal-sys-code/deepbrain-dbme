import json
import os

def load_data():
    with open("data/retention_sessions.json", "r") as f:
        sessions = json.load(f)
    
    with open("data/retention_queries.json", "r") as f:
        queries = json.load(f)
        
    # Index queries by trigger_session_id for faster lookup
    queries_by_session = {}
    for q in queries:
        sid = q["trigger_session_id"]
        if sid not in queries_by_session:
            queries_by_session[sid] = []
        queries_by_session[sid].append(q)
        
    return sessions, queries_by_session
