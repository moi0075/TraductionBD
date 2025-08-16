# Translation module
import ollama
import re

def ollama_llm(prompt,system_prompt="", model="gemma3n:e2b"):
    response = ollama.chat( 
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    # Remove any internal tags like <think> ... </think>
    response_content = response['message']['content']
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()

    return final_answer