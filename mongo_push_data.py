from pymongo import MongoClient
from mongoDB_cred import MONGO_URL

# Connect to your existing cluster
client = MongoClient(MONGO_URL)

# Choose a database under the same cluster
db = client["NetworkSecurity_db"]

# Choose a collection
collection = db["exceptions"]

# Insert sample data
collection.insert_one({
    "file": "exception.py",
    "line": 21,
    "error": "division by zero",
    "timestamp": "2026-07-01T21:09:00"
})

print("Data inserted successfully!")
