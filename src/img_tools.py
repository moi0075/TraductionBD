from PIL import Image
import numpy as np

def resize_image(img_path, max_side):
    """
    Redimensionne une image pour que son plus grand côté <= max_side.
    Retourne :
        - l'image redimensionnée en numpy.ndarray (RGB)
        - le facteur de réduction (ex: 2.0 signifie divisée par 2)
    """
    # Ouvrir et convertir en RGB
    img = Image.open(img_path).convert("RGB")
    
    # Dimensions originales
    w, h = img.size
    
    # Calcul du facteur de réduction
    scale = min(max_side / w, max_side / h, 1.0)
    
    # Nouvelle taille
    new_size = (int(w * scale), int(h * scale))
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Facteur : combien l'image a été réduite (w / new_w)
    reduction_factor = w / new_size[0]
    
    return np.array(img_resized), reduction_factor

def save_crops_from_coords(img_np, coords_list, output_folder):
    """
    Découpe et sauvegarde des zones d'une image à partir de coordonnées.

    Parameters:
    - img_np: image source sous forme de tableau numpy
    - coords_list: liste de tuples (x_min, y_min, x_max, y_max)
    - output_folder: dossier où sauvegarder les captures
    """
    img = Image.fromarray(img_np)
    
    for i, (x_min, y_min, x_max, y_max) in enumerate(coords_list):
        crop = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        crop_path = f"{output_folder}/cluster_{i}.png"
        crop.save(crop_path)
        print(f"Cluster {i} sauvegardé : {crop_path}")

