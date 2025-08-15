from PIL import Image
import numpy as np

def load_image_as_numpy(img_path, max_side=None):
    """
    Charge une image en numpy.ndarray (RGB), avec option de redimensionnement.
    Si max_side est None, l'image n'est pas redimensionnée.
    
    Returns:
        - numpy.ndarray : image RGB
        - float : facteur de réduction (1.0 si pas de resize)
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    if max_side is None:
        return np.array(img), 1.0

    scale = min(max_side / w, max_side / h, 1.0)
    new_size = (int(w * scale), int(h * scale))
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

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

