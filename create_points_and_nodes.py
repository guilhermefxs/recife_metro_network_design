import geopandas as gpd
import json
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, shape
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from tqdm import tqdm

# Load the GeoJSON data using the json module
geojson_path = 'zonas_com_frequencia.geojson'
with open(geojson_path, 'r') as f:
    geojson_data = json.load(f)

# Extract geometries and properties from the GeoJSON features
geometries = [shape(feature["geometry"]) for feature in geojson_data["features"]]
properties = [feature["properties"] for feature in geojson_data["features"]]

# Extract the bounding box of all geometries
bounds = unary_union(geometries).bounds
minx, miny, maxx, maxy = bounds

# Convert 400 meters to degrees approximately (for equatorial regions)
meters_per_degree = 111320  # Approx. meters per degree at the equator
distance_in_degrees = 400 / meters_per_degree

# Generate grid points
x_coords = np.arange(minx, maxx, distance_in_degrees)
y_coords = np.arange(miny, maxy, distance_in_degrees)
grid_points = [Point(x, y) for x in x_coords for y in y_coords]

# Filter points that are within the original geometries and assign path_coverage and zone
filtered_points = []
point_data = []
for point in tqdm(grid_points, desc="Filtering points"):
    for geom, prop in zip(geometries, properties):
        if geom.contains(point):
            point_data.append((point.x, point.y, prop['FREQUENCIA'], prop['ZONA']))
            filtered_points.append(point)
            break

# Create a list of coordinates and path_coverage from filtered points
point_coords = [(point.x, point.y) for point in filtered_points]

# Build a k-d tree for efficient neighbor search
tree = cKDTree(point_coords)

# Define the maximum distance for neighbors (400 meters in degrees)
max_distance = distance_in_degrees

# Find pairs of points within the maximum distance
edges = tree.query_pairs(max_distance)

# Convert pairs of indices to pairs of points
edges_points = []
for i, j in tqdm(edges, desc="Creating edges"):
    edges_points.append((filtered_points[i], filtered_points[j]))

# Prepare data for points and edges
points_data = [{'id': idx, 'x': x, 'y': y, 'path_coverage': path_coverage, 'zone': zone}
               for idx, (x, y, path_coverage, zone) in enumerate(point_data)]
edges_data = [{'source': i, 'target': j} for i, j in edges]

# Create DataFrames for points and edges
points_df = pd.DataFrame(points_data)
edges_df = pd.DataFrame(edges_data)

# Save points and edges to a CSV file
output_csv = 'grid.csv'
with open(output_csv, 'w') as f:
    f.write('# Points\n')
    points_df.to_csv(f, index=False)
    f.write('\n# Edges\n')
    edges_df.to_csv(f, index=False)

print(f"New CSV file created at: {output_csv}")
