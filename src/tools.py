
import os
import shutil
import time
import subprocess
from tqdm import tqdm
import psutil
import re

def clean_folder(folder_path):
    """
    Supprime tout le contenu d'un dossier sans supprimer le dossier lui-même.

    Args:
        folder_path (str): Chemin vers le dossier à nettoyer.
    """
    if not os.path.exists(folder_path):
        print(f"Le dossier {folder_path} n'existe pas.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # supprime fichier ou lien
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # supprime dossier et son contenu
        except Exception as e:
            print(f"Erreur lors de la suppression de {file_path}: {e}")

    print(f"Le dossier {folder_path} a été nettoyé avec succès.")

def launch_exe(path_to_exe, timeout=15):
    """
    Lance un fichier .exe et attend qu'il soit réellement lancé.

    Paramètres :
    - path_to_exe : str, chemin complet vers le fichier .exe
    - timeout : int, durée max (en secondes) avant d'abandonner

    Lève :
    - TimeoutError si l'application ne démarre pas à temps
    """
    if not os.path.isfile(path_to_exe):
        raise FileNotFoundError(f"Le fichier .exe n'a pas été trouvé : {path_to_exe}")

    exe_name = os.path.basename(path_to_exe)
    subprocess.Popen([path_to_exe], shell=True)
    print(f"Lancement de {exe_name}...")

    start_time = time.time()
    for _ in tqdm(range(timeout), desc="Démarrage", unit="s"):
        time.sleep(1)
        if any(proc.name().lower() == exe_name.lower() for proc in psutil.process_iter(['name'])):
            print(f"{exe_name} est maintenant lancé ✅")
            return
    raise TimeoutError(f"{exe_name} n'a pas démarré dans les {timeout} secondes.")

def natural_sort_key(s):
    # Sépare les nombres et le texte pour un tri naturel
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]