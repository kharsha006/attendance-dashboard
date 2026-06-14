# =============================================================
#  employee_db.py  —  PyMongo persistence layer for employee
#                     face embeddings (InsightFace ArcFace 512-d)
# =============================================================

import logging
from datetime import datetime, timezone
import numpy as np
from pymongo import MongoClient
import os

from config import MONGO_URI

log = logging.getLogger(__name__)

EMBEDDING_DIM = 512

class EmployeeDB:
    def __init__(self, db_uri: str = MONGO_URI):
        self.db_uri = db_uri
        self.client = MongoClient(self.db_uri)
        # Parse DB name from URI or fallback to attendance_db
        db_name = "attendance_db"
        try:
            parsed = self.db_uri.split("?")[-2].split("/")[-1]
            if parsed and parsed not in ["localhost:27017"]:
                db_name = parsed
        except:
            pass
        self.db = self.client[db_name]
        self.collection = self.db["employees"]

    def initialize(self):
        self.collection.create_index("employee_id", unique=True)
        log.debug("[EmployeeDB] Collection initialized at %s", self.db_uri)

    def upsert(
        self,
        employee_id: str,
        employee_name: str,
        embedding: np.ndarray,
        image_count: int,
    ):
        if len(embedding.shape) != 2 or embedding.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Embedding must be shape (N, {EMBEDDING_DIM}), got {embedding.shape}")

        ts = datetime.now(timezone.utc).isoformat()
        emb_list = embedding.astype(np.float32).flatten().tolist()
        
        self.collection.update_one(
            {"employee_id": employee_id},
            {"$set": {
                "employee_name": employee_name,
                "embedding": emb_list,
                "image_count": image_count,
                "enrollment_timestamp": ts
            }},
            upsert=True
        )
        log.info("[EmployeeDB] Upserted %s (%s) — %d images", employee_id, employee_name, image_count)

    def delete(self, employee_id: str) -> bool:
        res = self.collection.delete_one({"employee_id": employee_id})
        deleted = res.deleted_count > 0
        if deleted:
            log.info("[EmployeeDB] Deleted employee: %s", employee_id)
        else:
            log.warning("[EmployeeDB] Delete: employee not found: %s", employee_id)
        return deleted

    def get_all(self) -> dict:
        result = {}
        for doc in self.collection.find():
            arr = np.array(doc["embedding"], dtype=np.float32).reshape(-1, 512)
            result[doc["employee_id"]] = {
                "name":        doc.get("employee_name", ""),
                "embeddings":  list(arr),
                "image_count": doc.get("image_count", 0),
            }
        log.info("[EmployeeDB] Loaded %d employee(s)", len(result))
        return result

    def get_one(self, employee_id: str) -> dict | None:
        doc = self.collection.find_one({"employee_id": employee_id})
        if doc is None:
            return None
            
        arr = np.array(doc["embedding"], dtype=np.float32).reshape(-1, 512)
        return {
            "employee_id":          doc["employee_id"],
            "employee_name":        doc.get("employee_name", ""),
            "embedding":            list(arr),
            "image_count":          doc.get("image_count", 0),
            "enrollment_timestamp": doc.get("enrollment_timestamp", ""),
        }

    def list_employees(self) -> list:
        docs = self.collection.find({}, {"_id": 0, "employee_id": 1, "employee_name": 1, "image_count": 1, "enrollment_timestamp": 1}).sort("employee_name", 1)
        return list(docs)

    def count(self) -> int:
        return self.collection.count_documents({})

    def employee_exists(self, employee_id: str) -> bool:
        return self.collection.find_one({"employee_id": employee_id}, {"_id": 1}) is not None
