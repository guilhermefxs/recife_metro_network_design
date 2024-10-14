import pandas as pd
import networkx as nx
import folium
from folium import Popup
import json

class MetroNetworkDesign:
    def __init__(self, nodes_df=None, edges_df=None, coverage_threshold=None, target_zones=None):
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.coverage_threshold = coverage_threshold
        self.target_zones = target_zones
        self.graph = self.create_graph() if nodes_df is not None and edges_df is not None else None
        
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
        self.graph = self.create_graph()
        
        # Load paths
        with open(paths_filename, 'r') as f:
            paths_str_keys = json.load(f)
        
        # Convert string keys back to tuples
        paths = {(int(k.split('-')[0]), int(k.split('-')[1])): (v[0], v[1]) for k, v in paths_str_keys.items()}
        
        return paths
    
    def generate_map(self, paths, output_filename="metro_network_map.html"):
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
        
        m.save(output_filename)

# Example usage
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')

coverage_threshold = 0.01  # Adjust this threshold based on your dataset
target_zones = [173, 53, 24, 215, 59]  # Corresponding to T1, T2, T3, T4, T5

metro_network = MetroNetworkDesign(nodes_df, edges_df, coverage_threshold, target_zones)

# Find zone representatives based on highest path coverage
zone_representatives = metro_network.find_zone_representatives()

# Find the paths for all source-destination pairs between zones
paths = metro_network.algorithm_5(zone_representatives)

# Save results
metro_network.save_results(paths, 'nodes_saved.csv', 'paths_saved.json')

# Generate the map
metro_network.generate_map(paths)

# To reload the results later
# metro_network = MetroNetworkDesign()
# loaded_paths = metro_network.load_results('nodes_saved.csv', 'paths_saved.json')
# metro_network.generate_map(loaded_paths, "loaded_metro_network_map.html")
