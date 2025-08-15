# Main pipeline
from ocr import extract_text_from_image,ocr_results_to_dataframe, filter_by_score, cluster_polygons, add_cluster_column, bounding_boxes_by_cluster_with_text
from img_tools import save_crops_from_coords, resize_image
from traduction import ollama_translate_en_fr, translate_cluster_texts

def main(image):


    # Step 1: Extract text from image
    img_np, factor = resize_image(image, 2500)
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
    save_crops_from_coords(img_np, df_boxes[["x_min", "y_min", "x_max", "y_max"]].values, "outputs/ocr_outputs")

    # Step 8 : Remove text from img

    # Step 9 : Translate the text in each cluster
    df_translated = translate_cluster_texts(df_boxes, ollama_translate_en_fr, context="Translating dialogues from a webtoon", model="gemma3n:e4b")
    print(df_translated)
        
    # Step 10 : Write translation on img

if __name__ == '__main__':
    main("notebooks/ch_16_4.jpg")
