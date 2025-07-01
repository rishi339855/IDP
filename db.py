from pymongo import MongoClient
from pymongo.collection import Collection
from typing import Optional, Dict, Any, List

# Use the provided MongoDB URI
MONGO_URI = "mongodb://localhost:27017/IDP"
DB_NAME = "IDP"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
users_col: Collection = db["users"]
rides_col: Collection = db["rides"]
trips_col: Collection = db["trips"]

# --- USER OPERATIONS ---
def get_user(username: str) -> Optional[Dict[str, Any]]:
    return users_col.find_one({"username": username})

def create_user(user: Dict[str, Any]) -> None:
    users_col.insert_one(user)

def update_user(username: str, update: Dict[str, Any]) -> None:
    users_col.update_one({"username": username}, {"$set": update})

def get_all_drivers() -> List[Dict[str, Any]]:
    return list(users_col.find({"role": "driver"}))

def get_all_managers() -> List[Dict[str, Any]]:
    return list(users_col.find({"role": "manager"}))

def get_unassigned_drivers() -> List[Dict[str, Any]]:
    return list(users_col.find({"role": "driver", "fleet_manager": None}))

def assign_driver_to_manager(driver_username: str, manager_username: str) -> None:
    users_col.update_one({"username": driver_username}, {"$set": {"fleet_manager": manager_username}})

def get_drivers_for_manager(manager_username: str) -> List[Dict[str, Any]]:
    return list(users_col.find({"role": "driver", "fleet_manager": manager_username}))

# --- RIDE/EVENT OPERATIONS ---
def log_ride(event: Dict[str, Any]) -> None:
    rides_col.insert_one(event)

def get_rides_for_driver(driver_username: str) -> List[Dict[str, Any]]:
    return list(rides_col.find({"driver": driver_username}))

def get_all_rides() -> List[Dict[str, Any]]:
    return list(rides_col.find())

# --- TRIP OPERATIONS ---
def log_trip(trip: Dict[str, Any]) -> str:
    result = trips_col.insert_one(trip)
    return str(result.inserted_id)

def get_trips_for_driver(driver_username: str) -> List[Dict[str, Any]]:
    return list(trips_col.find({"driver": driver_username})) 