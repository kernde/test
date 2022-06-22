import folium
from folium.plugins import MarkerCluster
import pandas as pd
import webbrowser
from py2neo import Graph
from folium import plugins

graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))
# Load Nodes
nodes_company = graph.run(
    """
    MATCH (c:Company)
    where c.AddressZip = '1000'
    RETURN ID(c) as ID, c.NameShort as CompanyName, c.AddressLat as Lat, c.AddressLon as Lon, c.WorkersDirectorsNumber as WorkersDirectorsNumber,
                c.WorkersEmployeesNumber as WorkersEmployeesNumber, c.WorkersServantsNumber	as WorkersServantsNumber,
                c.WorkersWorkersNumber as WorkersWorkersNumber, c.degree as Degree, c.community as Community, c.rsr_proximity as Rsr_proximity
    """
).to_data_frame().set_index("ID")

nodes_company['Lat'] = pd.to_numeric(nodes_company['Lat'])
nodes_company['Lon'] = pd.to_numeric(nodes_company['Lon'])
nodes_company = nodes_company[nodes_company['Lon'].notna()] # about 4à comp
nodes_company = nodes_company.fillna(0)

coords = list(zip(nodes_company['Lat'],nodes_company['Lon'], nodes_company['CompanyName']))

#Create the map
my_map = folium.Map(location = coords[0], zoom_start = 13)

for lat, lng, name in coords[0:100]:
    folium.Marker([lat,lng], popup = name).add_to(my_map)

#Display the map
my_map.save('map_test.html')
webbrowser.open('output_files/map_test.html')

# heatmap test

map = folium.Map(location = [15,30], tiles='Cartodb dark_matter', zoom_start = 2)
# heat_data = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry ]

coords = list(zip(nodes_company['Lat'],nodes_company['Lon']))
plugins.HeatMap(coords, radius = 15, min_opacity=0.2).add_to(my_map)

# now specific RSR view
nodes_company = graph.run(
    """
    MATCH (c:Company)-[r:Involved_in]->(f:Fraud)
    RETURN ID(c) as ID, c.NameShort as CompanyName, c.AddressLat as Lat, c.AddressLon as Lon
    """
).to_data_frame().set_index("ID")
nodes_company = nodes_company[nodes_company['Lon'].notna()] # about 4à comp

coords = list(zip(nodes_company['Lat'],nodes_company['Lon']))
my_map = folium.Map(location = [51,4.5], zoom_start = 13)

plugins.HeatMap(coords, radius = 15, min_opacity=0.2).add_to(my_map)

my_map.save('map_test.html')
webbrowser.open('output_files/map_test.html')
import folium.plugins as fp
my_map = folium.Map(location = [50.84,4.35], zoom_start = 13) # focus BXL
callback = ("""function (row) {
                        var circle = L.circle(new L.LatLng(row[0], row[1]), {color: row[5] > 0.05 ? 'red':'green', 
                                                                             radius: row[4]>10 ? 10 : 5});
                        var popup = L.popup({maxWidth: '300'});
                        const comp= {name: row[2], community: row[3], degree: row[4], rsr_proximity: row[5]};
                        var mytext = $(`<div id='mytext' class='display_text' style='width: 100.0%; height: 100.0%;'> 
                        ${'<b>'+comp.name+'</b>'+
                          '<br><i>Community</i>:'+ comp.community+
                          '<br><i>Degree</i>:'+comp.degree+
                          '<br><i>Rsr proximity</i>:'+comp.rsr_proximity}</div>`)[0];
                        popup.setContent(mytext);
                        circle.bindPopup(popup);
                        return circle;
             };""")
coords = list(zip(nodes_company['Lat'],nodes_company['Lon'],
                  nodes_company['CompanyName'], nodes_company['Community'].astype(str),
                  nodes_company['Degree'].astype(str),
                  nodes_company['Rsr_proximity'].round(2).astype(str)))
FastMarkerCluster = plugins.FastMarkerCluster(coords, callback=callback).add_to(my_map)

my_map.save('map_test.html')
webbrowser.open('output_files/map_test.html')

### test graph now plot edges as well

graph_map = graph.run(
    """
    MATCH (c1:Company)-[o:OWNS]->(c2:Company)
    RETURN ID(o) as ID, c1.NameShort as CompanyName1, c2.NameShort as CompanyName2 , 
            c1.AddressLat as sLat, c1.AddressLon as sLon,c2.AddressLat as tLat, c2.AddressLon as tLon
    """
).to_data_frame() #.set_index("ID")
graph_map = graph_map.drop_duplicates() # multi ownership not relevant here
graph_map['sLat'] = pd.to_numeric(graph_map['sLat'])
graph_map['tLat'] = pd.to_numeric(graph_map['tLat'])
graph_map['sLon'] = pd.to_numeric(graph_map['sLon'])
graph_map['tLon'] = pd.to_numeric(graph_map['tLon'])
graph_map = graph_map[graph_map['sLon'].notna()]
graph_map = graph_map[graph_map['tLon'].notna()]

m = folium.Map([51,4.5], zoom_start=11)
for _, row in graph_map.iterrows():
    #folium.CircleMarker([row['sLat'], row['sLon']],
    #                    radius=2,
    #                    fill_color="blue", # divvy color
    #                   ).add_to(m)
    #folium.CircleMarker([row['tLat'], row['tLon']],
    #                    radius=2,
    #                    fill_color="blue", # divvy color
    #                   ).add_to(m)
    folium.PolyLine([[row['sLat'], row['sLon']],
                     [row['tLat'], row['tLon']]],weight=0.2, opacity=0.1).add_to(m)

m.save('graph_map.html')
webbrowser.open('output_files/graph_map.html')


# with model results an popover
nodes_company = graph.run(
    """
    MATCH (c:Company)
    RETURN ID(c) as ID, c.NameShort as CompanyName, c.AddressLat as Lat, c.AddressLon as Lon, c.WorkersDirectorsNumber as WorkersDirectorsNumber,
                c.WorkersEmployeesNumber as WorkersEmployeesNumber, c.WorkersServantsNumber	as WorkersServantsNumber,
                c.WorkersWorkersNumber as WorkersWorkersNumber, c.degree, c.community, c.rsr_proximity
                limit 100
    """
).to_data_frame().set_index("ID") # need to limit else the size of the html is huge
nodes_company = nodes_company[nodes_company['Lon'].notna()] # about 40 comp

my_map = folium.Map(location = [51,4.5], zoom_start = 13)
callback = ("""function (row) {
                        var circle = L.circle(new L.LatLng(row[0], row[1]), {color: 'red', radius: 5});
                        var popup = L.popup({maxWidth: '300'});
                        const display_text = {text: row[2]};
                        var mytext = $(`<div id='mytext' class='display_text' style='width: 100.0%; height: 100.0%;'> ${display_text.text}</div>
                        popup.setContent(mytext);
                        circle.bindPopup(popup);
                        return circle;
             };""")
coords = list(zip(nodes_company['Lat'],nodes_company['Lon'], nodes_company['CompanyName']))
plugins.FastMarkerCluster(coords, callback=callback).add_to(my_map)
my_map.save('map_test.html')
webbrowser.open('output_files/map_test.html')