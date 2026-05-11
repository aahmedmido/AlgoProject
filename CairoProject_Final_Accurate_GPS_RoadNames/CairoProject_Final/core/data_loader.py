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

        roads_data = self._read_json("roads.json")
        transport_data = self._read_json("transport.json")

        if isinstance(roads_data, dict):
            self.existing_roads = roads_data.get("existing_roads", roads_data.get("roads", []))
            self.new_roads = roads_data.get("new_roads", roads_data.get("potential_roads", []))
            self.traffic_patterns = roads_data.get("traffic_patterns", [])
        else:
            self.existing_roads = roads_data

        if isinstance(transport_data, dict):
            self.metro_lines = transport_data.get("metro_lines", [])
            self.bus_routes = transport_data.get("bus_routes", [])
            self.public_transport_demand = transport_data.get("public_transport_demand", [])
        else:
            self.bus_routes = transport_data

        return self