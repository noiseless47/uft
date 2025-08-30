from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TableSchema:
    required: List[str]
    dtypes: Dict[str, Any] | None = None

@dataclass
class DataSchema:
    matches: TableSchema
    squads: TableSchema
    events: TableSchema
    players: TableSchema
