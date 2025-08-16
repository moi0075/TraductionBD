# Scraping module à lacer pou créer des images
import os
import time
import requests
from bs4 import BeautifulSoup

def download_manhua(manhua_name, start_chapter, end_chapter, output_base="scans"):
    """
    Télécharge les chapitres d'un manhua depuis manhuatop.org.

    Args:
        manhua_name (str): Nom du manhua dans l'URL (ex: "records-of-the-swordsman-scholar").
        start_chapter (int): Chapitre de départ.
        end_chapter (int): Chapitre de fin.
        output_base (str): Dossier où seront stockés les chapitres.
    """
    base_url = "https://manhuatop.org/manhua"
    headers = {"User-Agent": "Mozilla/5.0"}

    for chap in range(start_chapter, end_chapter + 1):
        chapter_url = f"{base_url}/{manhua_name}/chapter-{chap}/"
        print(f"\n📖 Chapitre {chap} : {chapter_url}")

        response = requests.get(chapter_url, headers=headers)
        if response.status_code != 200:
            print(f"⚠️ Impossible d'accéder au chapitre {chap}, code {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        images = soup.select("img.wp-manga-chapter-img")

        if not images:
            print(f"⚠️ Aucun scan trouvé pour le chapitre {chap}")
            continue

        # Création du dossier du chapitre
        output_dir = os.path.join(output_base, manhua_name, f"chapitre_{chap}")
        os.makedirs(output_dir, exist_ok=True)

        for i, img in enumerate(images, start=1):
            img_url = img.get("data-src") or img.get("src")
            if not img_url:
                continue

            ext = img_url.split(".")[-1].split("?")[0]
            filename = os.path.join(output_dir, f"page_{i}.{ext}")

            print(f"Téléchargement : {img_url}")
            img_data = requests.get(img_url, headers=headers)
            img_data.raise_for_status()

            with open(filename, "wb") as f:
                f.write(img_data.content)

            time.sleep(1)  # Pause pour ne pas surcharger le serveur

        print(f"✅ Chapitre {chap} terminé. Images sauvegardées dans : {output_dir}")

# Exemple d'utilisation avec dossier personnalisé
download_manhua(
    manhua_name="records-of-the-swordsman-scholar",
    start_chapter=19,
    end_chapter=19,
    output_base="inputs//scans"
)

