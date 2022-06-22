import folium
from folium.plugins import MarkerCluster
import pandas as pd
import webbrowser
from py2neo import Graph
from folium import plugins, features
import geopandas as gpd
df = gpd.read_file("../Data/Integrated_DataFrame.geojson")

graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))
features.Choropleth(.geojson.json)
nil = gpd.read_file(fname)

# Load Nodes
nodes_company = graph.run(
    """
    MATCH (c:Company)
    RETURN ID(c) as ID, c.AddressLat as Lat, c.AddressLon as Lon
    """
).to_data_frame().set_index("ID")

nodes_company['Lat'] = pd.to_numeric(nodes_company['Lat'])
nodes_company['Lon'] = pd.to_numeric(nodes_company['Lon'])
nodes_company = nodes_company[nodes_company['Lon'].notna()] # about 40 comp
nodes_company = nodes_company.fillna(0) # they will be excluded of the map.

coords = list(zip(nodes_company['Lat'],nodes_company['Lon']))

# Node density visualisation (no connection)
map = folium.Map(location = [15,30], tiles='stamentoner', zoom_start = 2)

coords = list(zip(nodes_company['Lat'],nodes_company['Lon']))
plugins.HeatMap(coords, radius = 15, min_opacity=0.2).add_to(map)

#Display the map
map.save('output_files/map_company_node_density.html')
webbrowser.open('output_files/map_company_node_density.html')

# Now edges for (a) company graph and (b) dense company graph
graph_map_company = graph.run(
    """
    MATCH (c1:Company)-[o:OWNS]->(c2:Company)
    RETURN ID(o) as ID, c1.NameShort as CompanyName1, c2.NameShort as CompanyName2 , 
            c1.AddressLat as sLat, c1.AddressLon as sLon,c2.AddressLat as tLat, c2.AddressLon as tLon
    """
).to_data_frame()

graph_map_company = graph_map_company.drop_duplicates() # multi ownership not relevant here
graph_map_company['sLat'] = pd.to_numeric(graph_map_company['sLat'])
graph_map_company['tLat'] = pd.to_numeric(graph_map_company['tLat'])
graph_map_company['sLon'] = pd.to_numeric(graph_map_company['sLon'])
graph_map_company['tLon'] = pd.to_numeric(graph_map_company['tLon'])
graph_map = graph_map_company[graph_map_company['sLon'].notna()]
graph_map = graph_map[graph_map['tLon'].notna()]

m = folium.Map([51,4.5], tiles='stamentoner', zoom_start=11)
for _, row in graph_map.iterrows():
    folium.PolyLine([[row['sLat'], row['sLon']],
                     [row['tLat'], row['tLon']]],weight=0.2, opacity=0.1).add_to(m)

m.save('output_files/graph_map_edges_company.html')
webbrowser.open('output_files/graph_map_edges_company.html')

# working for edges
graph_map_company = graph.run(  # OPTIONAL MATCH (s:Company) -[:OWNS]-> (t:Company)
            """
            MATCH  (c1:Company) <- [:WORKS]- (w:Person) -[:WORKS]-> (c2:Company)
            RETURN ID(w) as ID,
            c1.AddressLat as sLat, c1.AddressLon as sLon,c2.AddressLat as tLat, c2.AddressLon as tLon
            """
).to_data_frame()

graph_map_company = graph_map_company.drop_duplicates() # multi ownership not relevant here
graph_map_company['sLat'] = pd.to_numeric(graph_map_company['sLat'])
graph_map_company['tLat'] = pd.to_numeric(graph_map_company['tLat'])
graph_map_company['sLon'] = pd.to_numeric(graph_map_company['sLon'])
graph_map_company['tLon'] = pd.to_numeric(graph_map_company['tLon'])
graph_map_company = graph_map_company[graph_map_company['sLon'].notna()]
graph_map_company = graph_map_company[graph_map_company['tLon'].notna()]

# sample for rendering map size (only for visual purposes).
graph_map_company = graph_map_company.sample(n=100)

m = folium.Map([51,4.5], tiles='stamentoner', zoom_start=11)
for _, row in graph_map_company.iterrows():
    folium.PolyLine([[row['sLat'], row['sLon']],
                     [row['tLat'], row['tLon']]],weight=1, opacity=0.2, color="red").add_to(m)

m.save('output_files/graph_map_edges_company_work.html')
webbrowser.open('output_files/graph_map_edges_company_work.html')




