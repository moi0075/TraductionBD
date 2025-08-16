# OCR extraction 
from paddleocr import PaddleOCR
import pandas as pd
from shapely.geometry import Polygon
import networkx as nx

def extract_text_from_image(image_path, ocr):
     # model fr mieux pour l'anglais pas logique mais marche mieux...
    result = ocr.predict(
        input=image_path)
    
    return result

def ocr_results_to_dataframe(result):
    texts = result[0]["rec_texts"]
    polys = result[0]["rec_polys"]
    scores = result[0]["rec_scores"]
    
    data = []
    for i in range(len(texts)):
        if len(polys[i]) != 4:
            raise ValueError(f"Le polygone de la ligne {i} a {len(polys[i])} points au lieu de 4")
        
        row = {"text": texts[i], "score": scores[i]}
        for j, point in enumerate(polys[i]):
            row[f"x{j+1}"] = point[0]
            row[f"y{j+1}"] = point[1]
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def filter_by_score(df, min_score=0.5):
    df_filtered = df[df['score'] >= min_score].reset_index(drop=True)
    return df_filtered

def cluster_polygons(df, *coord_cols, margin_factor=0.1):

    if len(coord_cols) % 2 != 0:
        raise ValueError("Il faut un nombre pair de colonnes pour former les points (x, y).")
    
    n_points = len(coord_cols) // 2
    polys_points = []
    
    for idx, row in df.iterrows():
        points = []
        for i in range(n_points):
            x_col = coord_cols[2*i]
            y_col = coord_cols[2*i + 1]
            points.append([row[x_col], row[y_col]])
        polys_points.append(points)
    
    polygons = [Polygon(p) for p in polys_points]
    
    # Agrandir chaque polygone proportionnellement à sa taille
    polygons_expanded = []
    for poly in polygons:
        size = poly.area**0.5
        margin = size * margin_factor
        polygons_expanded.append(poly.buffer(margin))
    
    # Construire le graphe
    G = nx.Graph()
    G.add_nodes_from(range(len(polygons_expanded)))
    
    for i in range(len(polygons_expanded)):
        for j in range(i+1, len(polygons_expanded)):
            if polygons_expanded[i].intersects(polygons_expanded[j]):
                G.add_edge(i, j)
    
    clusters = list(nx.connected_components(G))
    return clusters

def add_cluster_column(df, clusters):
    df_copy = df.copy()
    cluster_col = [-1] * len(df_copy)  # valeur par défaut

    for cluster_idx, cluster in enumerate(clusters):
        for row_idx in cluster:
            cluster_col[row_idx] = cluster_idx

    df_copy['cluster'] = cluster_col
    return df_copy


def bounding_boxes_by_cluster_with_text(df):
    """
    Pour chaque cluster, crée un rectangle englobant et concatène les textes.
    
    Parameters:
    - df : DataFrame avec colonnes ['text','x1','y1','x2','y2','x3','y3','x4','y4','cluster']
    
    Returns:
    - df_boxes : DataFrame avec ['cluster', 'x_min', 'y_min', 'x_max', 'y_max', 'text']
    """
    clusters = df['cluster'].unique()
    data = []

    for clus in clusters:
        df_clus = df[df['cluster'] == clus]
        # Récupérer toutes les coordonnées x et y
        xs = df_clus[['x1','x2','x3','x4']].values.flatten()
        ys = df_clus[['y1','y2','y3','y4']].values.flatten()
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Concaténer tous les textes du cluster en les séparant par un espace
        cluster_text = " ".join(df_clus['text'].astype(str).tolist())
        
        data.append({
            'cluster': clus,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'text': cluster_text
        })
    
    df_boxes = pd.DataFrame(data)
    return df_boxes
