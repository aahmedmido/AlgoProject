from pathlib import Path
from typing import Dict, List, Any
import json
import networkx as nx


class CairoTransportationData:

    def __init__(self, data_dir: str | Path = "data") -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = base_dir / data_dir

        self.neighborhoods: List[Dict[str, Any]] = []
        self.facilities: List[Dict[str, Any]] = []
        self.existing_roads: List[Dict[str, Any]] = []
        self.new_roads: List[Dict[str, Any]] = []
        self.traffic_patterns: List[Dict[str, Any]] = []
        self.metro_lines: List[Dict[str, Any]] = []
        self.bus_routes: List[Dict[str, Any]] = []
        self.public_transport_demand: List[Dict[str, Any]] = []

        self.graph = nx.Graph()

    def _read_json(self, filename: str):

        path = self.data_dir / filename

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def load(self):

        self.neighborhoods = self._read_json("neighborhoods.json")
        self.facilities = self._read_json("facilities.json")
        self.existing_roads = self._read_json("existing_roads.json")
        self.new_roads = self._read_json("new_roads.json")
        self.traffic_patterns = self._read_json("traffic_patterns.json")
        self.metro_lines = self._read_json("metro_lines.json")
        self.bus_routes = self._read_json("bus_routes.json")
        self.public_transport_demand = self._read_json(
            "public_transport_demand.json"
        )

        return self