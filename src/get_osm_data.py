import osmnx as ox
import overpy
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import random
import os

import branca.colormap as cm

from sklearn.cluster import DBSCAN # version: 0.24.1
from collections import Counter


def gen_poi_data(api, amenity_mapping):
    # iterate over all categories, call the overpass api,
    # and add the results to the poi_data list
    # to test queries can use the site https://overpass-turbo.eu/
    poi_data = []
    for idx, (amenity_cat, amenity) in enumerate(amenity_mapping):
        query = f"""node["{amenity_cat}"="{amenity}"]({bbox});out;"""
        result = api.query(query)
        # print(amenity, len(result.nodes))

        for node in result.nodes:
            data = {}
            name = node.tags.get('name', 'N/A')
            data['name'] = name
            data['amenity'] = amenity_cat + '__' + amenity
            data['geometry'] = Point(node.lon, node.lat)
            poi_data.append(data)

    return poi_data


def plot_poi_map(gdf_poi):
    f, ax = plt.subplots(1,1,figsize=(10,10))
    admin.plot(ax=ax, color = 'none', edgecolor = 'k', linewidth = 2)
    gdf_poi.plot(column = 'amenity', ax=ax, legend = True, alpha = 0.3)


def gen_color_dict(gdf, column, colors):
    color_dict = {}
    unique_items = gdf[column].unique()
    for i, item in enumerate(unique_items):
        color_dict[item] = colors[i % len(colors)]
    return color_dict


def gen_unique_color_dict(item_list):
    color_dict = {
        item: "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for item in item_list}
    return color_dict


def centroid_from_admin_geometry(admin):
    # get the centroid of the city and set up the map
    x, y = admin.geometry.to_list()[0].centroid.xy
    centroid = [y[0], x[0]]
    return centroid


def centroid_from_gdf(gdf):
    # get the centroid from gdf
    min_longitude, min_latitude, max_longitude, max_latitude = gdf.total_bounds
    centroid = [(min_latitude + max_latitude) / 2, (min_longitude + max_longitude) / 2]
    return centroid


def gen_folium_map(centroid, gdf_poi, column='amenity', cluster_map=False, zoom_start=12,
                   savepath='./html', savename='folium_map.html'):

    m = folium.Map(location=centroid, zoom_start=zoom_start, tiles='Cartodb Positron')

    if cluster_map:
        # get unique, random colors for each cluster
        unique_clusters = gdf_poi['cluster_id'].unique()
        color_dict = gen_unique_color_dict(unique_clusters)
    else:
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 'gray', 'cyan', 'magenta', 'yellow', 'lightblue',
                  'lime']
        # transform the gdf_poi
        color_dict = gen_color_dict(gdf_poi, column, colors)

    # visualize the pois with a scatter plot
    for idx, row in gdf_poi.iterrows():
        amenity = row['amenity']
        lat = row['geometry'].y
        lon = row['geometry'].x
        if cluster_map:
            cluster_id = row['cluster_id']
            color = color_dict[cluster_id]
        else:
            color = color_dict.get(amenity, 'gray')  # default to gray if not in the colormap

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,  # No transparency for dot markers
            popup=amenity,
        ).add_to(m)

    m.save(os.path.join(savepath, savename))


def apply_dbscan_clustering(gdf_poi, eps=50, min_samples=1):
    """eps: The max dist between two samples for one to be considered as in the neighborhood of the other
    """

    feature_matrix = gdf_poi['geometry'].apply(lambda geom: (geom.x, geom.y)).tolist()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # You can adjust min_samples as needed
    cluster_labels = dbscan.fit_predict(feature_matrix)
    gdf_poi['cluster_id'] = cluster_labels

    return gdf_poi


def plot_cluster_size_histogram(gdf):
    clusters = gdf.cluster_id.to_list()
    clusters_cnt = Counter(clusters).most_common()

    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist([cnt for c, cnt in clusters_cnt], bins=20)
    ax.set_yscale('log')
    ax.set_xlabel('Cluster size', fontsize=14)
    ax.set_ylabel('Number of clusters', fontsize=14)
    plt.show()


if __name__ == "__main__":

    # select the city of interest and get the admin polygon
    borough = 'Ealing'
    city = 'London'
    country = 'UK'

    admin = ox.geocode_to_gdf(city)
    # admin.plot()

    # start the api
    api = overpy.Overpass()

    # get the enclosing bounding box
    # we can define the coordinate reference system. here we use EPSG:4326 used by gps and get min and max x and y
    # then join together into a bounding box string
    minx, miny, maxx, maxy = admin.to_crs(crs=4326).bounds.T[0]
    bbox = ','.join([str(miny), str(minx), str(maxy), str(maxx)])

    # define the OSM categories of interest
    amenity_mapping = [
        ("amenity", "cafe"),
        ("tourism", "gallery"),
        ("amenity", "pub"),
        ("amenity", "bar"),
        ("amenity", "marketplace"),
        ("sport", "yoga"),
        ("amenity", "studio"),
        ("shop", "music"),
        ("shop", "second_hand"),
        ("amenity", "foodtruck"),
        ("amenity", "music_venue"),
        ("shop", "books"),
    ]

    poi_data = gen_poi_data(api, amenity_mapping)

    # transform the results into a geodataframe
    gdf_poi = gpd.GeoDataFrame(poi_data)
    print(len(gdf_poi))

    plot_poi_map(gdf_poi)

    # get spatial overlay with gathered data and crop with city polygon
    gdf_poi = gpd.overlay(gdf_poi, admin[['geometry']])
    gdf_poi.crs = 4326
    print(len(gdf_poi))

    plot_poi_map(gdf_poi)

    centroid = centroid_from_admin_geometry(admin)

    # gen_folium_html_map(admin, gdf_poi, column='amenity')

    gen_folium_map(centroid, gdf_poi, column='amenity', cluster_map=False, savename='poi_map.html')

    # do the clustering

    # # transforming to local crs
    gdf_poi_filt = gdf_poi.to_crs(27700)

    # do the clustering
    # can adjust eps_value and maybe min_samples=1
    eps_value = 50
    # min_samples 2 gives strange result - essentially all non-clustered single points get cluster -1
    # so could perhaps exclude these to avoid filtering clusters with n or more points
    min_samples = 1
    clustered_gdf_poi = apply_dbscan_clustering(gdf_poi_filt, eps_value, min_samples=min_samples)
    # clustered_gdf_poi = apply_dbscan_clustering(gdf_poi, eps_value)

    # Print the GeoDataFrame with cluster IDs
    print(f'Number of clusters found: {len(set(clustered_gdf_poi.cluster_id))}')

    plot_cluster_size_histogram(clustered_gdf_poi)

    clusters = clustered_gdf_poi.cluster_id.to_list()

    # only keep clusters with x or more points of interest
    to_keep = [c for c, cnt in Counter(clusters).most_common() if cnt >= 5]
    clustered_gdf_poi = clustered_gdf_poi[clustered_gdf_poi.cluster_id.isin(to_keep)]
    clustered_gdf_poi = clustered_gdf_poi.to_crs(4326)
    print(f'Number of clusters kept: {len(to_keep)}')

    # get the centroid of the city and set up the map
    city_centroid = centroid_from_gdf(clustered_gdf_poi)

    gen_folium_map(centroid, clustered_gdf_poi, column='amenity', cluster_map=True, savename='poi_cluster_map.html')

    print()
