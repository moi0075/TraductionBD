# Translation module
import ollama
import re

def ollama_translate_en_fr(text, context="", all_dialogues="", model="gemma3n:e2b"):
    """
    Translate English text into French using Ollama.
    
    Parameters:
    - text: str, English text to translate
    - context: str, optional contextual instructions
    - model: str, Ollama model name
    
    Returns:
    - str: translated text in French
    """
    prompt = (
        f"Translate the following text from English to French.\n"
        f"Context: {context}\n"
        f"All dialogues : {all_dialogues}\n"
        f"Instruction: ONLY output the translated French text in UPPERCASE, naturally and fluently, as dialogue. Remove or ignore any OCR artifacts like /, #, or other errors. Do not add explanations, notes, line breaks, or extra formatting."
        f"If you absolutely do not know how to translate a word or phrase, do not translate it and leave it as-is.\n"
        f"Text: {text}"
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional English-to-French translator. Translate the text naturally and fluently as spoken dialogue in a comic or webtoon. Ignore any OCR artifacts or incorrect characters."},
            {"role": "user", "content": prompt}
        ]
    )

    # Remove any internal tags like <think> ... </think>
    response_content = response['message']['content']
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()

    return final_answer

def translate_cluster_texts(df_boxes, translator_func, context="Translating dialogues from a webtoon", model="gemma3n:e2b"):
    """
    Traduire le texte de chaque cluster à partir d'un DataFrame df_boxes
    et ajouter une colonne 'translated'.
    
    Parameters:
    - df_boxes : DataFrame avec colonnes ['cluster', 'x_min','y_min','x_max','y_max','text']
    - translator_func : fonction de traduction, prenant text, context et previous_dialogues
    - context : str, instructions contextuelles pour le traducteur
    
    Returns:
    - df_result : DataFrame identique à df_boxes avec une colonne 'translated'
    """
    df_result = df_boxes.copy()
    all_dialogues = df_result['text'].str.cat(sep=' ')

    translated_texts = []

    for idx, row in df_result.iterrows():
        cluster_text = row["text"]
        print(f"\nCluster {row['cluster']} original text:\n{cluster_text}\n")

        # Traduire
        translated_text = translator_func(
            text=cluster_text,
            context=context,
            all_dialogues=all_dialogues,
            model=model
        )

        print(f"Cluster {row['cluster']} translated:\n{translated_text}\n")
        translated_texts.append(translated_text)

    # Ajouter la colonne 'translated'
    df_result["translated"] = translated_texts
    return df_result

