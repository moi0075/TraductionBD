import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Union, Tuple

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


def save_crops_from_coords(img_np, coords_list, output_folder, scale=1.0):
    """
    Découpe et sauvegarde des zones d'une image à partir de coordonnées,
    avec possibilité d'agrandir ou réduire chaque crop via un facteur scale.

    Parameters:
    - img_np: image source sous forme de tableau numpy
    - coords_list: liste de tuples (x_min, y_min, x_max, y_max)
    - output_folder: dossier où sauvegarder les captures
    - scale: facteur d'agrandissement/reduction des crops (1.0 = taille originale)
    """
    img = Image.fromarray(img_np)
    
    for i, (x_min, y_min, x_max, y_max) in enumerate(coords_list):
        # Calcul du centre du crop
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min) * scale
        h = (y_max - y_min) * scale

        # Calcul des nouvelles coordonnées
        new_x_min = max(0, cx - w / 2)
        new_y_min = max(0, cy - h / 2)
        new_x_max = min(img.width, cx + w / 2)
        new_y_max = min(img.height, cy + h / 2)

        crop = img.crop((int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)))
        crop_path = f"{output_folder}/cluster_{i}.png"
        crop.save(crop_path)
        print(f"Cluster {i} sauvegardé : {crop_path}")

def average_grayscale(img_path):
    """
    Calcule la moyenne de gris d'une image.

    Parameters:
    - img_path : chemin vers l'image

    Returns:
    - float : valeur moyenne de gris (0 = noir, 255 = blanc)
    """
    # Charger l'image et la convertir en niveaux de gris
    img = Image.open(img_path).convert("L")  # "L" = grayscale
    
    # Convertir en tableau numpy
    img_np = np.array(img, dtype=np.float32)
    
    # Calculer la moyenne
    return img_np.mean()

def create_and_save_solid_image(width, height, color=(255, 255, 255), save_path="image.png"):
    """
    Crée une image unie et la sauvegarde directement.

    Parameters:
    - width : largeur de l'image
    - height : hauteur de l'image
    - color : tuple RGB, par défaut blanc (255, 255, 255)
    - save_path : chemin où sauvegarder l'image

    Returns:
    - PIL.Image.Image : image créée
    """
    img = Image.new("RGB", (width, height), color)
    img.save(save_path)
    print(f"✅ Image sauvegardée ici : {save_path}")
    return img

def get_image_size(image_path):
    """
    Retourne la largeur et la hauteur d'une image.

    Parameters:
    - image_path : chemin vers l'image

    Returns:
    - (width, height) : tuple de largeur et hauteur
    """
    img = Image.open(image_path)
    return img.size  # img.size renvoie (width, height)

def draw_centered_text(
    image_path: str, 
    text: str, 
    font_path: str, 
    font_size: int, 
    output_path: str, 
    margin: int = 10, 
    min_font_size: int = 1, 
    fill_color: Union[Tuple[int,int,int], str] = (0,0,0),
    line_spacing_percent: float = 20.0
):
    """
    Écrit du texte centré sur une image en s'assurant que le texte tient parfaitement.
    
    - image_path: chemin de l'image source
    - text: texte (peut contenir '\n')
    - font_path: chemin vers un .ttf
    - font_size: taille de départ (la fonction réduira si nécessaire)
    - output_path: chemin où sauvegarder
    - margin: marge en pixels autour de la zone texte
    - min_font_size: taille minimale autorisée
    - fill_color: couleur du texte (tuple RGB ou nom)
    - line_spacing_percent: espacement entre les lignes en % de la hauteur de police
                           (100% = hauteur de police, 50% = moitié, 0% = lignes qui se touchent)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image introuvable: {image_path}")
    if not os.path.isfile(font_path):
        raise FileNotFoundError(f"Police introuvable: {font_path}")

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # zone disponible
    max_width = max(1, img.width - 2 * margin)
    max_height = max(1, img.height - 2 * margin)

    def split_text_lines_for_font(font):
        """Retourne la liste de lignes adaptées à max_width, en cassant les mots si besoin."""
        lines = []
        for para in text.split("\n"):
            words = para.split()
            if not words:
                lines.append("")  # ligne vide
                continue
            
            current = ""
            for word in words:
                candidate = current + (" " if current else "") + word
                bbox = draw.textbbox((0,0), candidate, font=font)
                w = bbox[2] - bbox[0]
                
                if w <= max_width:
                    current = candidate
                else:
                    # si current non vide -> pousser current puis start with word (or break word)
                    if current:
                        lines.append(current)
                    
                    # now handle word which might be too long by itself
                    # if word alone is too wide, break by characters
                    single_bbox = draw.textbbox((0,0), word, font=font)
                    single_w = single_bbox[2] - single_bbox[0]
                    
                    if single_w <= max_width:
                        current = word
                    else:
                        # break word into chunks of characters
                        chunk = ""
                        for ch in word:
                            test_chunk = chunk + ch
                            bbox_ch = draw.textbbox((0,0), test_chunk, font=font)
                            if bbox_ch[2] - bbox_ch[0] <= max_width:
                                chunk = test_chunk
                            else:
                                if chunk:
                                    lines.append(chunk)
                                chunk = ch
                        current = chunk
            
            if current:
                lines.append(current)
        
        return lines

    def calculate_line_metrics(font):
        """Calcule les métriques de ligne avec l'interligne personnalisé."""
        try:
            ascent, descent = font.getmetrics()
            base_line_height = ascent + descent
        except Exception:
            # fallback : bbox of "Ay"
            bbox_ay = draw.textbbox((0,0), "Ay", font=font)
            base_line_height = bbox_ay[3] - bbox_ay[1]
        
        # Calcul de l'interligne en fonction du pourcentage
        line_spacing = int(base_line_height * (line_spacing_percent / 100.0))
        total_line_height = base_line_height + line_spacing
        
        return base_line_height, line_spacing, total_line_height

    # Essayer les tailles à partir de font_size (descendre si nécessaire)
    chosen_font = None
    chosen_lines = None
    chosen_line_height = None
    chosen_base_height = None
    chosen_spacing = None

    size = font_size
    while size >= min_font_size:
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception as e:
            # police non chargée pour cette taille (rare) -> décrémente
            size -= 1
            continue

        lines = split_text_lines_for_font(font)
        if not lines:
            # rien à écrire -> ok
            chosen_font = font
            chosen_lines = [""]
            chosen_base_height, chosen_spacing, chosen_line_height = calculate_line_metrics(font)
            break

        # Calcul des métriques avec interligne personnalisé
        base_height, spacing, line_height = calculate_line_metrics(font)
        
        # Hauteur totale : on compte (n-1) interlignes pour n lignes
        if len(lines) > 1:
            total_height = len(lines) * base_height + (len(lines) - 1) * spacing
        else:
            total_height = base_height

        # largeur maximale réelle
        max_line_w = 0
        for ln in lines:
            bbox_ln = draw.textbbox((0,0), ln, font=font)
            w_ln = bbox_ln[2] - bbox_ln[0]
            if w_ln > max_line_w:
                max_line_w = w_ln

        # vérifier si tout rentre
        if total_height <= max_height and max_line_w <= max_width:
            chosen_font = font
            chosen_lines = lines
            chosen_line_height = line_height
            chosen_base_height = base_height
            chosen_spacing = spacing
            break

        size -= 1

    # Si aucune taille convenable trouvée -> utiliser min_font_size
    if chosen_font is None:
        chosen_font = ImageFont.truetype(font_path, max(min_font_size, 8))
        chosen_lines = split_text_lines_for_font(chosen_font)
        chosen_base_height, chosen_spacing, chosen_line_height = calculate_line_metrics(chosen_font)

    # Calculer origine (avec marges) pour centrer verticalement
    if len(chosen_lines) > 1:
        total_text_height = len(chosen_lines) * chosen_base_height + (len(chosen_lines) - 1) * chosen_spacing
    else:
        total_text_height = chosen_base_height if chosen_lines else 0
    
    y_start = margin + (max_height - total_text_height) // 2

    # Dessiner ligne par ligne, centrée horizontalement dans la zone
    y = int(y_start)
    for i, ln in enumerate(chosen_lines):
        bbox_ln = draw.textbbox((0,0), ln, font=chosen_font)
        w_ln = bbox_ln[2] - bbox_ln[0]
        x = margin + (max_width - w_ln) // 2
        
        draw.text((int(x), int(y)), ln, font=chosen_font, fill=fill_color)
        
        # Pour la ligne suivante, ajouter hauteur de base + espacement
        # (sauf pour la dernière ligne)
        if i < len(chosen_lines) - 1:
            y += chosen_base_height + chosen_spacing

    # Sauvegarder
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    img.save(output_path)

    # Retour utile : la taille de police choisie et infos
    print(f"Image sauvegardée : {output_path}")
    print(f"Font size used: {chosen_font.size}")
    print(f"Line spacing: {line_spacing_percent}% ({chosen_spacing}px)")
    
    return {
        "font_size": chosen_font.size, 
        "lines": chosen_lines, 
        "base_line_height": chosen_base_height,
        "line_spacing": chosen_spacing,
        "total_line_height": chosen_line_height,
        "line_spacing_percent": line_spacing_percent
    }

def paste_image(img1_path, img2_path, x, y, save_path=None):
    """
    Colle img2 sur img1 aux coordonnées (x, y).

    Parameters:
        img1_path (str): Chemin vers l'image de fond.
        img2_path (str): Chemin vers l'image à coller.
        x (int): Coordonnée x où coller l'image.
        y (int): Coordonnée y où coller l'image.
        save_path (str, optional): Chemin pour sauvegarder le résultat. Si None, retourne l'image PIL.

    Returns:
        Image: Objet PIL.Image si save_path est None.
    """
    # Ouvrir les images
    img1 = Image.open(img1_path).convert("RGBA")
    img2 = Image.open(img2_path).convert("RGBA")

    # Coller img2 sur img1 aux coordonnées (x, y)
    img1.paste(img2, (x, y), img2)  # le 3e param permet de gérer la transparence

    if save_path:
        img1.save(save_path)
        print(f"Image saved to {save_path}")
    else:
        return img1
