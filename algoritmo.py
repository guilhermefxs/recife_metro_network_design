import pandas as pd
import networkx as nx
import folium
from folium import Popup
import json
import math
import os

class MetroNetworkDesign:
    def __init__(self, nodes_df=None, edges_df=None, coverage_threshold=None, target_zones=None, graph_filename="metro_graph.graphml"):
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.coverage_threshold = coverage_threshold
        self.target_zones = target_zones
        self.graph_filename = graph_filename
        
        # Load or create the graph
        if os.path.exists(self.graph_filename):
            self.graph = self.load_graph()
        else:
            self.graph = self.create_graph()
            self.save_graph()
    
    def create_graph(self):
        graph = nx.Graph()
        for _, node in self.nodes_df.iterrows():
            graph.add_node(node['id'], path_coverage=node['path_coverage'], zone=node['zone'], x=node['x'], y=node['y'])
        for _, edge in self.edges_df.iterrows():
            source = edge['source']
            target = edge['target']
            weight = self.calculate_weight(source, target)
            graph.add_edge(source, target, weight=weight)
        return graph
    
    def calculate_weight(self, source, target):
        source_coverage = self.nodes_df[self.nodes_df['id'] == source]['path_coverage'].values[0]
        target_coverage = self.nodes_df[self.nodes_df['id'] == target]['path_coverage'].values[0]
        return 1 / min(source_coverage, target_coverage)
    
    def save_graph(self):
        nx.write_graphml(self.graph, self.graph_filename)
        print(f"Graph saved to {self.graph_filename}")
    
    def load_graph(self):
        graph = nx.read_graphml(self.graph_filename)
        # Convert node IDs from string/float to integers
        graph = nx.relabel_nodes(graph, {n: int(float(n)) for n in graph.nodes})
        print(f"Graph loaded from {self.graph_filename}")
        return graph
    
    def find_zone_representatives(self):
        zone_representatives = {}
        for zone in self.target_zones:
            # Find a node with the highest coverage in the zone to represent it
            zone_nodes = self.nodes_df[self.nodes_df['zone'] == zone]
            representative = zone_nodes.loc[zone_nodes['path_coverage'].idxmax()]['id']
            zone_representatives[zone] = representative
        return zone_representatives
    
    def algorithm_5(self, zone_representatives):
        best_paths = {}
        
        for zone_src, src_node in zone_representatives.items():
            for zone_dest, dest_node in zone_representatives.items():
                if zone_src != zone_dest:
                    try:
                        path = nx.shortest_path(self.graph, source=src_node, target=dest_node, weight='weight')
                        total_coverage = sum(1 / self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                        best_paths[(zone_src, zone_dest)] = (path, total_coverage)
                        # Print the path and its total coverage
                        print(f"Path from Zone {zone_src} to Zone {zone_dest}: {path} | Coverage: {total_coverage:.4f}")
                    except nx.NetworkXNoPath:
                        best_paths[(zone_src, zone_dest)] = ([], 0)
                        print(f"No path found from Zone {zone_src} to Zone {zone_dest}")
        
        return best_paths
    
    def save_results(self, paths, nodes_filename, paths_filename):
        # Save nodes information
        self.nodes_df.to_csv(nodes_filename, index=False)

        # Convert tuple keys to strings and convert numpy.int64 to int
        paths_str_keys = {f"{zone_src}-{zone_dest}": ([int(node) for node in path], coverage) 
                            for (zone_src, zone_dest), (path, coverage) in paths.items()}

        # Save paths to JSON
        with open(paths_filename, 'w') as f:
            json.dump(paths_str_keys, f)
    
    def load_results(self, nodes_filename, paths_filename):
        # Load nodes information
        self.nodes_df = pd.read_csv(nodes_filename)
        
        # Load paths
        with open(paths_filename, 'r') as f:
            paths_str_keys = json.load(f)
        
        # Convert string keys back to tuples
        paths = {(int(k.split('-')[0]), int(k.split('-')[1])): (v[0], v[1]) for k, v in paths_str_keys.items()}
        
        return paths
    
    def generate_map(self, paths, output_filename="metro_network_map.html", highlight_path=None):
        m = folium.Map(location=[self.nodes_df['y'].mean(), self.nodes_df['x'].mean()], zoom_start=13)
        
        # Gather all nodes and edges involved in the paths
        nodes_in_paths = set()
        edges_in_paths = []
        
        for (zone_src, zone_dest), (path, coverage) in paths.items():
            if path:
                nodes_in_paths.update(path)
                edges_in_paths.extend([(path[i], path[i + 1]) for i in range(len(path) - 1)])
        
        # Plot only the nodes involved in the paths
        for node_id in nodes_in_paths:
            node = self.nodes_df[self.nodes_df['id'] == node_id].iloc[0]
            folium.CircleMarker(
                location=[node['y'], node['x']],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                popup=Popup(f"Node ID: {node['id']}<br>Zone: {node['zone']}<br>Coverage: {node['path_coverage']}"),
            ).add_to(m)
        
        # Plot the edges from the solution paths
        for src, tgt in edges_in_paths:
            src_node = self.nodes_df[self.nodes_df['id'] == src].iloc[0]
            tgt_node = self.nodes_df[self.nodes_df['id'] == tgt].iloc[0]
            folium.PolyLine(
                locations=[(src_node['y'], src_node['x']), (tgt_node['y'], tgt_node['x'])],
                color='red',
                weight=3,
                opacity=0.7,
            ).add_to(m)
        
        # Highlight the specific path if provided
        if highlight_path:
            for i in range(len(highlight_path) - 1):
                src = highlight_path[i]
                tgt = highlight_path[i + 1]
                src_node = self.nodes_df[self.nodes_df['id'] == src].iloc[0]
                tgt_node = self.nodes_df[self.nodes_df['id'] == tgt].iloc[0]
                folium.PolyLine(
                    locations=[(src_node['y'], src_node['x']), (tgt_node['y'], tgt_node['x'])],
                    color='green',
                    weight=5,
                    opacity=1,
                ).add_to(m)
        
        m.save(output_filename)
    
    def find_shortest_path(self, start_zone, end_zone):
        # Find the nearest node within the start zone
        start_node = self.find_nearest_node_in_zone(start_zone)
        if start_node is None:
            print(f"No nodes found in Zone {start_zone}.")
            return [], 0
        
        # Find the nearest node within the end zone
        end_node = self.find_nearest_node_in_zone(end_zone)
        if end_node is None:
            print(f"No nodes found in Zone {end_zone}.")
            return [], 0
        
        try:
            # Calculate the shortest path between the nearest nodes in the zones
            path = nx.shortest_path(self.graph, source=start_node, target=end_node, weight='weight')
            distance = nx.shortest_path_length(self.graph, source=start_node, target=end_node, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return [], 0

    def find_nearest_node_in_zone(self, zone):
        """
        Finds the nearest node within a specific zone.
        
        Args:
            zone (int): Zone ID.
        
        Returns:
            int: ID of the nearest node in the zone, or None if no nodes are found.
        """
        # Filter nodes in the specified zone
        zone_nodes = self.nodes_df[self.nodes_df['zone'] == zone]
        if zone_nodes.empty:
            return None
        
        # Return the node with the highest coverage (or any other criteria)
        return zone_nodes.loc[zone_nodes['path_coverage'].idxmax()]['id']
    
    def haversine(self, lat1, lon1, lat2, lon2):
        # Radius of the Earth in kilometers
        R = 6371.0
        
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        
        # Difference in coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # Distance in kilometers
        distance = R * c
        return distance
    
    def calculate_real_distance(self, path):
        total_distance = 0
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            src_node = self.nodes_df[self.nodes_df['id'] == src].iloc[0]
            tgt_node = self.nodes_df[self.nodes_df['id'] == tgt].iloc[0]
            # Calculate Haversine distance
            distance = self.haversine(src_node['y'], src_node['x'], tgt_node['y'], tgt_node['x'])
            total_distance += distance
        return total_distance

# Example usage
nodes_df = pd.read_csv('./nodes.csv')
edges_df = pd.read_csv('./edges.csv')

coverage_threshold = 0.01  # Adjust this threshold based on your dataset
target_zones = [173, 53, 24, 215, 59]  # Corresponding to T1, T2, T3, T4, T5

# Initialize the MetroNetworkDesign with a graph filename
metro_network = MetroNetworkDesign(nodes_df, edges_df, coverage_threshold, target_zones, graph_filename="metro_graph.graphml")

# Find zone representatives based on highest path coverage
zone_representatives = metro_network.find_zone_representatives()

# Find the paths for all source-destination pairs between zones
paths = metro_network.algorithm_5(zone_representatives)

# Save results
metro_network.save_results(paths, 'nodes_saved.csv', 'paths_saved.json')

# Find the shortest path between two specific zones
start_zone = 173
end_zone = 53
shortest_path, distance = metro_network.find_shortest_path(start_zone, end_zone)
print(f"Shortest path from Zone {start_zone} to Zone {end_zone}: {shortest_path} | Distance: {distance:.4f}")

# Calculate the real distance of the shortest path using Haversine
real_distance = metro_network.calculate_real_distance(shortest_path)
print(f"Real distance of the shortest path: {real_distance:.4f} km")

# Generate the map with the shortest path highlighted
metro_network.generate_map(paths, f"metro_network_map_path_{start_zone}_{end_zone}.html", highlight_path=shortest_path)
metro_network.generate_map(paths, f"metro_network_map.html")
# To reload the results later
# metro_network = MetroNetworkDesign(graph_filename="metro_graph.graphml")
# loaded_paths = metro_network.load_results('nodes_saved.csv', 'paths_saved.json')
# metro_network.generate_map(loaded_paths, "loaded_metro_network_map.html")
