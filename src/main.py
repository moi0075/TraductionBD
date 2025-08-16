# Main pipeline
from ocr import extract_text_from_image,ocr_results_to_dataframe, filter_by_score, cluster_polygons, add_cluster_column, bounding_boxes_by_cluster_with_text
from img_tools import get_image_size, save_crops_from_coords, load_image_as_numpy,create_and_save_solid_image,average_grayscale, draw_centered_text
from traduction import ollama_translate_en_fr, translate_cluster_texts
from tools import clean_folder, launch_exe, natural_sort_key
import os

def main(image):

    # Step 1: Extract text from image
    img_np, factor = load_image_as_numpy(image, None)
    result = extract_text_from_image(img_np)

    # Step 2: Convert OCR results to DataFrame
    df = ocr_results_to_dataframe(result)

    # Step 3: Filter DataFrame by score
    filtered_df = filter_by_score(df, min_score=0.7)

    # Step 4 : Cluster the polygons
    clusters = cluster_polygons(filtered_df, "x1","y1","x2","y2","x3","y3","x4","y4", margin_factor=0.1)

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
    df_translated = translate_cluster_texts(df_boxes, ollama_translate_en_fr, context="Translating dialogues from a webtoon", model="gemma3:12b")
    df_translated['translated_upper'] = df_translated['translated'].str.upper()
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

if __name__ == '__main__':
    launch_exe(r"C:\Users\teo\AppData\Local\Programs\Ollama\ollama app.exe", timeout=10)

    clean_folder("outputs/ocr_outputs")
    clean_folder("outputs/text_remove_outputs")
    clean_folder("outputs/text_drawn_outputs")

    main("notebooks/ch_0_2.jpg")

    
