from pathlib import Path
from typing import Dict, List, Any
import json
import networkx as nx


class CairoTransportationData:

    def __init__(self, data_dir: str | Path = "data") -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = base_dir / data_dir

        self.neighborhoods = []
        self.facilities = []
        self.existing_roads = []
        self.new_roads = []
        self.traffic_patterns = []
        self.metro_lines = []
        self.bus_routes = []
        self.public_transport_demand = []

        self.graph = nx.Graph()

    def _read_json(self, filename: str):
        path = self.data_dir / filename

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def load(self):

        self.neighborhoods = self._read_json("neighborhoods.json")
        self.facilities = self._read_json("facilities.json")

        roads = self._read_json("roads.json")
        transport = self._read_json("transport.json")

        self.existing_roads = roads.get("existing_roads", roads.get("roads", []))
        self.new_roads = roads.get("new_roads", [])
        self.traffic_patterns = roads.get("traffic_patterns", [])

        self.metro_lines = transport.get("metro_lines", [])
        self.bus_routes = transport.get("bus_routes", [])
        self.public_transport_demand = transport.get(
            "public_transport_demand", []
        )

        self._build_graph()

        return self

    def _build_graph(self):

        for item in self.neighborhoods:
            self.graph.add_node(
                item["id"],
                label=item["name"],
                type="neighborhood"
            )

        for item in self.facilities:
            self.graph.add_node(
                item["id"],
                label=item["name"],
                type="facility"
            )

        for road in self.existing_roads:

            source = road["from"]
            target = road["to"]

            weight = road.get("distance", 1)

            self.graph.add_edge(
                source,
                target,
                weight=weight
            )

    def locations_for_ui(self):

        result = []

        for node_id, attrs in self.graph.nodes(data=True):

            label = attrs.get("label", node_id)

            result.append((node_id, label))

        return sorted(result, key=lambda x: x[1])

    def get_node_name(self, node):

        if node in self.graph.nodes:

            return self.graph.nodes[node].get(
                "label",
                node
            )

        return node