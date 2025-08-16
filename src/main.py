# Main pipeline
from ocr import extract_text_from_image,ocr_results_to_dataframe, filter_by_score, cluster_polygons, add_cluster_column, bounding_boxes_by_cluster_with_text
from img_tools import get_image_size, save_crops_from_coords, load_image_as_numpy,create_and_save_solid_image,average_grayscale, draw_centered_text, paste_image
from traduction import ollama_llm
from tools import clean_folder, launch_exe, natural_sort_key
import os
import json
from PIL import Image
from pathlib import Path
from paddleocr import PaddleOCR
import pandas as pd
import shutil

def main(image, ocr):

    max_attempts = 3
    df = None
    for attempt in range(1, max_attempts + 1):
        print(f"🔄 OCR attempt {attempt} for {image}")

        # Step 1: Extract text from image
        img_np, factor = load_image_as_numpy(image, None)
        result = extract_text_from_image(img_np, ocr)

        # Step 2: Convert OCR results to DataFrame
        df = ocr_results_to_dataframe(result)

        # Step 3: Filter DataFrame by score
        filtered_df = filter_by_score(df, min_score=0.7)    
    
        if not filtered_df.empty:
            print(f"✅ OCR successful on attempt {attempt}")
            break  # On sort dès qu'on a des résultats
        else:
            print(f"⚠️ No text detected on attempt {attempt}")
            if attempt == max_attempts:
                # Après 3 tentatives, copier l'image telle quelle
                img_text_drawn_outputs = os.path.join(
                    "outputs", "translated_chapter",
                    os.path.splitext(os.path.relpath(file_path, "inputs/scans"))[0] + ".png"
                )
                os.makedirs(os.path.dirname(img_text_drawn_outputs), exist_ok=True)
                img = Image.open(image)
                img.save(img_text_drawn_outputs, format="PNG")
                print(f"📄 Image copiée sans OCR dans {img_text_drawn_outputs}")
                return

    # Step 4 : Cluster the polygons
    clusters = cluster_polygons(filtered_df, "x1","y1","x2","y2","x3","y3","x4","y4", margin_factor=0.2)

    # Step 5 : Add cluster information to the DataFrame
    clustered_df = add_cluster_column(filtered_df, clusters)
    print(clustered_df)

    # Step 6 : Get bounding boxes for each cluster
    df_boxes = bounding_boxes_by_cluster_with_text(clustered_df)

    # Step 7 : Save crops from bounding boxes
    ocr_outputs_path = "outputs/ocr_outputs"
    save_crops_from_coords(img_np, df_boxes[["x_min", "y_min", "x_max", "y_max"]].values, ocr_outputs_path,1)

    # Step 8 : Remove text from img
    text_remove_path = "outputs/text_remove_outputs"
    for filename in os.listdir(ocr_outputs_path):
        img_path = os.path.join(ocr_outputs_path, filename)
        size = get_image_size(img_path)
        if average_grayscale(img_path) > 255/2:
            create_and_save_solid_image(size[0], size[1], color=(255, 255, 255), save_path=os.path.join(text_remove_path, filename))
        else:
            create_and_save_solid_image(size[0], size[1], color=(0, 0, 0), save_path=os.path.join(text_remove_path, filename))

    # Step 9 : Translate the text in each cluster gemma3n:e2b gemma3:12b
    prompt = """You translate English phrases into French in a natural and fluent style for webtoon dialogue.
    Instruction: ONLY output a list containing the French translations of the given texts, in natural and fluent dialogue. Do not add explanations, notes, line breaks, or extra formatting. Remove or ignore any OCR artifacts like /, #, or other errors. If you absolutely do not know how to translate a word or phrase, leave it as-is. """+json.dumps(df_boxes["text"].tolist(), ensure_ascii=False)
    system_prompt = f"You translate English phrases into French in a natural and fluent style for webtoon dialogue. ONLY output a list containing the French translations of the given texts. The list must have the same length as the original list, that is, {len(df_boxes['text'])} elements."

    l_translated_upper = [] 
    while len(l_translated_upper)!=len(df_boxes['text']) : # On vérifie que l'on a bien une traduction pour chaque élément
        r = ollama_llm(prompt,system_prompt, model="gemma3:12b")
        l_translated = json.loads(r)
        l_translated_upper = [s.upper() for s in l_translated]
        print(l_translated_upper)

    df_boxes['translated_upper'] = l_translated_upper
    df_translated = df_boxes.copy()
    print(df_translated)

    # Step 10 : Write translation on img
    text_drawn_outputs = "outputs/text_drawn_outputs"
    files = sorted(os.listdir(text_remove_path), key=natural_sort_key)
    for i, filename in enumerate(files):
        img_path = os.path.join(text_remove_path, filename)
        text = df_translated["translated_upper"][i]
        out_path = os.path.join(text_drawn_outputs, filename)

        if average_grayscale(img_path) > 255/2:
            fill_color=(0, 0, 0)
        else:
            fill_color=(255, 255, 255)

        draw_centered_text(
            image_path=img_path,
            text=text,
            font_path="inputs\\fonts\\Komika Text-FontZillion\\Fonts\\komtxtb_.ttf",
            font_size=1000,
            output_path=out_path,
            margin=2,
            fill_color=fill_color,
            line_spacing_percent=-20
        )

    # Step 11 : Paste the translated text images onto the original image
    img_text_drawn_outputs = os.path.join("outputs", "translated_chapter",os.path.splitext(os.path.relpath(file_path, "inputs/scans"))[0] + ".png")
    os.makedirs(os.path.dirname(img_text_drawn_outputs), exist_ok=True)# Création du dossier parent si nécessaire
    img = Image.open(image)
    img.save(img_text_drawn_outputs, format="PNG")

    for index, row in df_translated.iterrows():
        x_min = row['x_min']
        y_min = row['y_min']
        cluster_id = row['cluster']
        file_name = f"cluster_{cluster_id}.png"
        print(f"x_min: {x_min}, y_min: {y_min}, file_name: {file_name}")
        paste_image(img_text_drawn_outputs, text_drawn_outputs+"\\"+file_name, x_min, y_min, save_path=img_text_drawn_outputs)

if __name__ == '__main__':
    launch_exe(r"C:\Users\teo\AppData\Local\Programs\Ollama\ollama app.exe", timeout=1)

    ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='fr')
    
    base_dir = "inputs//scans"
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                file_path = os.path.join(root, file)
                print(file_path)

                clean_folder("outputs/ocr_outputs")
                clean_folder("outputs/text_remove_outputs")
                clean_folder("outputs/text_drawn_outputs")
                main(file_path,ocr)

