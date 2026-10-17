"""Semantic layer: metric definitions with roles and measurement periods."""
from pathlib import Path

import yaml

DEFAULT = Path(__file__).parent / "semantic_layer" / "sim_metrics.yaml"


class SemanticLayer:
    def __init__(self, path=DEFAULT):
        self.metrics = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["metrics"]
        self.by_column = {m["column"]: m for m in self.metrics}

    def lookup(self, query):
        q = query.lower()
        hits = [m for m in self.metrics
                if q in m["name"].lower() or q == m["column"].lower()
                or any(q in a.lower() for a in m.get("aliases", []))]
        return hits

    def columns(self, role):
        return [m["column"] for m in self.metrics if m["role"] == role]

    def period(self, column):
        m = self.by_column.get(column)
        return None if m is None else m["period"]
