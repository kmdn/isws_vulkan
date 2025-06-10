import os
import csv
from transformers import pipeline


def get_english_sentences(filepath):
    """ Reads English sentences from CSV file (sentences.csv).
    Args:
        filepath (str): Path to the file containing English sentences.
    Returns:
        list: List of English sentences.
    """

    sentences = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Ignore header
        next(reader)
        for row in reader:
            if row:
                sentences.append(row[1])
    return sentences

def translate_with_mistral(english_text):
    """
    Translates English text to Italian using Mistral.
    
    Args:
        english_text (str): The English text to translate.
    Returns:
        str: The translated Italian text.
    """
    # Load Mistral pipeline with chat template support
    pipe = pipeline(
        "text-generation",
        model="unsloth/llama-3-8b",#"mistralai/Mistral-7B-v0.1",#"meta-llama/Meta-Llama-3-8B-Instruct",# #
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Create chat messages for better instruction following
    # Load prompt from prompt_translation.txt
    with open('prompt_translation.txt', 'r', encoding='utf-8') as f:
        prompt = f.read().strip()

    messages = [
        {"role": "system", "content": "You are an Italian expert on art history and a professional English-Italian translator. Given an English text from Giorgio Vasari's \"Lives of The Artists\" (Original title: Le vite de' più eccellenti pittori, scultori e architettori), provide only the original Italian text without any explanations or additional text."},
        {"role": "user", "content": english_text}#f"{prompt}:\n\n{english_text}"}
    ]
    
    # Generate response
    response = pipe(
        messages,
        max_new_tokens=2000,
        temperature=0.1,  # Low temperature for consistent translations
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    
    # Extract the assistant's response
    italian_translation = response[0]['generated_text'][-1]['content'].strip()
    
    return italian_translation


def translate_with_llama3(english_text):
    """
    Translates English text to Italian using LLAMA3.
    
    Args:
        english_text (str): The English text to translate.
    Returns:
        str: The translated Italian text.
    """
    # Load LLAMA3 pipeline with chat template support
    pipe = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",#"unsloth/llama-3-8b", #
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Create chat messages for better instruction following
    # Load prompt from prompt_translation.txt
    with open('prompt_translation.txt', 'r', encoding='utf-8') as f:
        prompt = f.read().strip()

    messages = [
        {"role": "system", "content": "You are an Italian expert on art history and a professional English-Italian translator. Given an English text from Giorgio Vasari's \"Lives of The Artists\" (Original title: Le vite de' più eccellenti pittori, scultori e architettori), provide only the original Italian text without any explanations or additional text."},
        {"role": "user", "content": english_text}#f"{prompt}:\n\n{english_text}"}
    ]
    
    # Generate response
    response = pipe(
        messages,
        max_new_tokens=2000,
        temperature=0.1,  # Low temperature for consistent translations
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    
    # Extract the assistant's response
    italian_translation = response[0]['generated_text'][-1]['content'].strip()
    
    return italian_translation

def translate_with_helsinki_nlp(english_text):
    # Load the translation pipeline
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-it")

    english_text = english_text[:512]

    # Translate the text
    translated_text = translator(english_text, max_length=512)[0]['translation_text']
    return translated_text



def translate_english_to_italian(english_text, variant=2):
    """ Map English text to Italian text by using translations or LLMs.
    Args:
        english_text (str): The English text to translate.
    Returns:
        str: The translated Italian text.
    """
    match variant:
        case 0:
            translated_text = translate_with_helsinki_nlp(english_text)
            return translated_text
        case 1:
            # "Translate" the text with LLAMA3 
            return translate_with_llama3(english_text)
        case 2:
            # "Translate" the text with Mistral
            translated_text = translate_with_mistral(english_text)
            return translated_text
        case _:
            raise ValueError("Invalid translation variant specified. Use 0 for Helsinki-NLP, 1 for LLAMA3, or 2 for Mistral.")
    
    return translated_text


def substring_search_original_translation(text1, text2, min_length=1):
    """
    Finds all maximized common substrings between two texts.
    A maximized substring is one that cannot be extended further while remaining common.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
    
    Returns:
        list: List of tuples containing (substring, start_pos_text1, start_pos_text2, length)
    """
    def find_all_common_substrings(s1, s2, min_length=1):
        """Find all common substrings with their positions."""
        common_substrings = []
        
        for i in range(len(s1)):
            for j in range(len(s2)):
                # Find the longest common substring starting at positions i and j
                length = 0
                while (i + length < len(s1) and 
                       j + length < len(s2) and 
                       s1[i + length] == s2[j + length]):
                    length += 1
                
                if length >= min_length:
                    substring = s1[i:i + length]
                    common_substrings.append((substring, i, j, length))
        
        return common_substrings
    
    def get_maximized_substrings(common_substrings):
        """Filter to keep only maximized substrings."""
        maximized = []
        
        # Sort by length (descending) to process longer substrings first
        sorted_substrings = sorted(common_substrings, key=lambda x: x[3], reverse=True)
        
        for current in sorted_substrings:
            current_str, current_i, current_j, current_len = current
            is_maximized = True
            
            # Check if this substring is contained in any already added substring
            for existing in maximized:
                existing_str, existing_i, existing_j, existing_len = existing
                
                # Check if current substring is contained within existing one
                if (existing_i <= current_i <= existing_i + existing_len - current_len and
                    existing_j <= current_j <= existing_j + existing_len - current_len and
                    current_str in existing_str):
                    is_maximized = False
                    break
            
            if is_maximized:
                # Check if this substring can be extended
                can_extend = False
                for other in sorted_substrings:
                    other_str, other_i, other_j, other_len = other
                    if (other_len > current_len and
                        other_i <= current_i < other_i + other_len and
                        other_j <= current_j < other_j + other_len and
                        current_str in other_str):
                        can_extend = True
                        break
                
                if not can_extend:
                    maximized.append(current)
        
        return maximized
    
    # Convert to lowercase for case-insensitive comparison
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    # Find all common substrings
    all_common = find_all_common_substrings(text1_lower, text2_lower, min_length=min_length)
    
    # Get maximized substrings
    maximized_substrings = get_maximized_substrings(all_common)
    
    # Sort by position in first text
    maximized_substrings.sort(key=lambda x: x[1])
    
    # Return the list of maximized substrings
    return maximized_substrings

def analyze_translation_similarity(english_text, italian_text, min_length):
    """
    Analyzes similarity between original and translated text using maximized substrings.
    
    Args:
        english_text (str): Original English text
        italian_text (str): Translated Italian text
    
    Returns:
        dict: Analysis results including maximized substrings and similarity metrics
    """
    maximized_substrings = substring_search_original_translation(english_text, italian_text, min_length)
    
    # Calculate similarity metrics
    total_chars_english = len(english_text)
    total_chars_italian = len(italian_text)
    
    common_chars = sum(length for _, _, _, length in maximized_substrings)
    
    similarity_ratio = common_chars / max(total_chars_english, total_chars_italian) if max(total_chars_english, total_chars_italian) > 0 else 0
    
    return {
        'maximized_substrings': maximized_substrings,
        'common_characters': common_chars,
        'english_length': total_chars_english,
        'italian_length': total_chars_italian,
        'similarity_ratio': similarity_ratio
    }



if __name__ == "__main__":

    WORKING_DIR = '/mnt/webscistorage/wf7467/isws/isws_vulkan/'
    os.chdir(WORKING_DIR)
    DATA_DIR = './data/'
    MAXIMAL_SUBSTRING_SEARCH = True
    ANALYSE_TRANSLATED_SIMILARITY = True

    english_sentences = get_english_sentences(DATA_DIR + 'sentences.csv')

    english_text = english_sentences[0]
    italian_translation = translate_english_to_italian(english_text)

    print(f"English: {english_text}")
    print(f"Italian: {italian_translation}")


    if MAXIMAL_SUBSTRING_SEARCH:
        print("\n--- Substring Analysis ---")
        
        # Example with two similar texts
        original_italian_text = ""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "A quick brown fox jumps over a lazy dog"
        # Minimum length for substrings to be considered
        min_length = 5
        
        results = substring_search_original_translation(text1, text2, min_length=min_length)
        print(f"\nComparing:\nText 1: '{text1}'\nText 2: '{text2}'")
        print("\nMaximized common substrings:")
        for substring, pos1, pos2, length in results:
            print(f"  '{substring}' (length: {length}) - Text1[{pos1}:{pos1+length}], Text2[{pos2}:{pos2+length}]")
    

    if ANALYSE_TRANSLATED_SIMILARITY:
        # Analyze translation similarity
        print("\n--- Translation Analysis ---")
        analysis = analyze_translation_similarity(english_text, italian_translation, min_length)
        print(f"Similarity ratio: {analysis['similarity_ratio']:.2%}")
        print(f"Common characters: {analysis['common_characters']}")
        print("Maximized substrings in translation:")
        for substring, pos1, pos2, length in analysis['maximized_substrings']:
            print(f"  '{substring}' (length: {length})")