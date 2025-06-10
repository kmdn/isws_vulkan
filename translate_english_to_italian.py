# Get all the data 

def translate_english_to_italian(english_text):
    """ Translates English text to Italian by using LLAMA3 as a LLM.
    Args:
        english_text (str): The English text to translate.
    Returns:
        str: The translated Italian text.
    """
    from transformers import pipeline

    # Load the translation pipeline
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-it")

    # Translate the text
    translated_text = translator(english_text, max_length=400)[0]['translation_text']
    
    return translated_text

if __name__ == "__main__":
    # Example usage
    english_text = "Hello, how are you?"
    italian_translation = translate_english_to_italian(english_text)
    print(f"English: {english_text}")
    print(f"Italian: {italian_translation}")