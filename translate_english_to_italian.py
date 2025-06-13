import os
import json
import csv
from transformers import pipeline
from openai import OpenAI
from thefuzz import fuzz, process
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

#try:
#    nltk.data.find('corpora/stopwords')
#except LookupError:
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


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
        model="mistralai/Mistral-7B-v0.1",#"unsloth/llama-3-8b",#"mistralai/Mistral-7B-v0.1",#"meta-llama/Meta-Llama-3-8B-Instruct",# #
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


def translate_with_deepseek(english_text, prompt_file='./prompt_translation.txt', out_response_path='./response.txt'):
    # Load API key from file
    api_key_file = ".deepseek_api_key.txt"
    with open(api_key_file, "r") as f:
        DEEPSEEK_API_KEY = f.read().strip()

    # Init LLM client
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        #messages=[
        #    {"role": "system", "content": "You are a helpful assistant, an expert at generating new mentions for entity linking tasks."},
        #    {"role": "user", "content": prompt},
        #],
        messages = [
                {"role": "system", "content": "You are an Italian expert on art history and a professional English-Italian translator. Given an English text from Giorgio Vasari's \"Lives of The Artists\" (Original title: Le vite de' più eccellenti pittori, scultori e architettori), provide only the original Italian text without any explanations or additional text."},
                {"role": "user", "content": english_text}#f"{prompt}:\n\n{english_text}"}
            ],
            stream=False,
            max_tokens=8000
        )
    return response.choices[0].message.content.strip()


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


def load_json_cache(filepath="./translation_cache.json"):
    """ Loads the JSON cache file if it exists.
    Returns:
        dict: The loaded JSON data or an empty dictionary if the file does not exist.
    """
    cache_file = filepath
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def add_to_cache(cache, english_text, italian_translation, replace_if_exists=False, save=True):
    entry = cache.get(english_text, None)

    if entry is None or replace_if_exists:
        cache[english_text] = italian_translation

    if save:
        save_json_cache(cache, filepath="./translation_cache.json")

def save_json_cache(cache, filepath="./translation_cache.json"):
    """ Saves the JSON cache to a file.
    Args:
        cache (dict): The cache data to save.
        filepath (str): The path to the cache file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
    print(f"Cache saved to {filepath}")

def cache_check(cache, english_text):
    """ Checks if the English text is already in the cache.
    Args:
        cache (dict): The cache data.
        english_text (str): The English text to check.
    Returns:
        str: The Italian translation if found, None otherwise.
    """
    return cache.get(english_text, None)

def translate_english_to_italian(english_text, variant=3, cache=load_json_cache()):
    """ We call it "translate", but actually it's a semi-translate and mostly map 
        English text to Italian text by using translations or LLMs.
    Args:
        english_text (str): The English text to translate.
    Returns:
        str: The translated Italian text.
    """
    cache = load_json_cache()
    translated_text = cache_check(cache=cache, english_text=english_text)
    if translated_text is not None:
        print(f"Cache hit: {english_text[0:min(50, len(english_text))]}...")
        return translated_text
    
    match variant:
        case 0:
            translated_text = translate_with_helsinki_nlp(english_text)
            add_to_cache(cache, english_text, translated_text)
            return translated_text
        case 1:
            # "Translate" the text with LLAMA3 
            translated_text = translate_with_llama3(english_text)
            add_to_cache(cache, english_text, translated_text)
            return translated_text
        case 2:
            # "Translate" the text with Mistral
            translated_text = translate_with_mistral(english_text)
            add_to_cache(cache, english_text, translated_text)
            return translated_text
        case 3:
            translated_text = translate_with_deepseek(english_text, prompt_file='./prompt_translation.txt')
            add_to_cache(cache, english_text, translated_text)
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

def get_italian_sentences(filepath):
    """ Reads Italian sentences from a TXT file.
    Args:
        filepath (str): Path to the file containing Italian sentences.
    Returns:
        list: List of Italian sentences.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        italian_sentences = file.readlines()
    
    # Clean up sentences
    italian_sentences = [sentence.strip() for sentence in italian_sentences if sentence.strip()]
    
    return italian_sentences


def remove_italian_stopwords(text):
    """
    Removes Italian stopwords from the given text.
    
    Args:
        text (str): Italian text to process
    
    Returns:
        str: Text with stopwords removed
    """
    # Get Italian stopwords
    italian_stopwords = set(stopwords.words('italian'))
    
    # Tokenize the text
    words = word_tokenize(text.lower(), language='italian')
    
    # Remove stopwords and punctuation
    filtered_words = [
        word for word in words 
        if word not in italian_stopwords and word not in string.punctuation and word.isalpha()
    ]
    
    return ' '.join(filtered_words)

def remove_italian_stopwords_list(text_list):
    """
    Removes Italian stopwords from a list of texts.
    
    Args:
        text_list (list): List of Italian texts
    
    Returns:
        list: List of texts with stopwords removed
    """
    return [remove_italian_stopwords(text) for text in text_list]


def clean_text(text):
    ret_text = remove_italian_stopwords(italian_translation).strip()
    re.sub(' +', ' ', ret_text)
    return ret_text


if __name__ == "__main__":
    print("Starting translation process...")
    WORKING_DIR = '/mnt/webscistorage/wf7467/isws/isws_vulkan/'
    os.chdir(WORKING_DIR)

    DATA_DIR = './data/'

    print("Loading English sentences from CSV file...")
    english_sentences = get_english_sentences(DATA_DIR + 'sentences.csv')
    # Get the original Italian sentences from the TXT file found at
    # https://it.wikisource.org/wiki/Le_vite_de%27_pi%C3%B9_eccellenti_pittori,_scultori_e_architettori_%281568%29
    print("Loading original Italian sentences from TXT file...")
    original_italian_sentences = get_italian_sentences(DATA_DIR + "Le_vite_de'_più_eccellenti_pittori,_scultori_e_architettori_(1568).txt")



    translations = []
    print("Loading translation cache...")
    #cache = load_json_cache()

    print("Starting translation of English sentences to Italian...")
    for i in range(len(english_sentences)):
        english_text = english_sentences[i]
        italian_translation = translate_english_to_italian(english_text)
        translations.append(italian_translation)
        #save_json_cache(cache, filepath="./translation_cache.json")
        print(f"English: {english_text}")
        print(f"Italian: {italian_translation}")



    # We have the translation, so now we can compare it with the original Italian sentences
    print("\n--- Comparing translations with original Italian sentences ---")
    # Which top longest maximized substrings to show
    SHOW_TOP = 3
    MAXIMAL_SUBSTRING_SEARCH = False
    ANALYSE_TRANSLATED_SIMILARITY = False
    FUZZY_MATCHING = True  # Use fuzzy matching to find similar sentences

    if MAXIMAL_SUBSTRING_SEARCH:
        for original_italian_text in original_italian_sentences:
            min_length = 10
            maximal_length = 0
            maximal_length_substring = ""

            for italian_translation in translations:
                italian_translation_clean = clean_text(italian_translation)
                original_italian_text_clean = clean_text(original_italian_text)
                results = substring_search_original_translation(original_italian_text_clean, italian_translation_clean, min_length=min_length)
                print(f"\nComparing:\nOriginal: '{original_italian_text_clean}'\nTranslation: '{italian_translation_clean}'")
                print("\nMaximized common substrings:")

                # Sort results by length in descending order
                results = sorted(results, key=lambda x: x[3], reverse=True)

                for substring, pos1, pos2, length in results[:SHOW_TOP]:
                    print(f"  '{substring}' (length: {length}) - Text1[{pos1}:{pos1+length}], Text2[{pos2}:{pos2+length}]")
    
                if ANALYSE_TRANSLATED_SIMILARITY:
                    # Analyze translation similarity
                    print("\n--- Translation Analysis ---")
                    analysis = analyze_translation_similarity(original_italian_text, italian_translation, min_length)
                    print(f"Similarity ratio: {analysis['similarity_ratio']:.2%}")
                    print(f"Common characters: {analysis['common_characters']}")
                    print("Maximized substrings in translation:")
                    for substring, pos1, pos2, length in analysis['maximized_substrings']:
                        if length > maximal_length:
                            maximal_length = length
                            maximal_length_substring = substring
                        #print(f"  '{substring}' (length: {length})")
                    print(f"Longest maximized substring: '{maximal_length_substring}' (length: {maximal_length})")

                # Fuzzy matching approach


    if FUZZY_MATCHING:
        original_italian_sentences_clean = []
        for original_sentence in original_italian_sentences:
            original_sentence_clean = clean_text(original_sentence)
            original_italian_sentences_clean.append(original_sentence_clean)

        original_italian_sentences_clean_split_by_sentences = []
        for original_clean_line in original_italian_sentences_clean:
            original_clean_line_split_by_sentences = original_clean_line.split(".")
            original_italian_sentences_clean_split_by_sentences.append(original_clean_line_split_by_sentences)

        FUZZY_MATCH_BY_SPLIT_SENTENCES = True

        if FUZZY_MATCH_BY_SPLIT_SENTENCES:
            # For every document
            for i, italian_translation in enumerate(translations):
                match_results = []
                # For every original Italian sentence

                # Translation sentences
                for trans_split_sentence in italian_translation_clean:
                    # We submatch by sentences.
                    english_text = english_sentences[i]  # Get corresponding English text
                    italian_translation_clean = clean_text(italian_translation).split(".")

                    # Original sentences
                    orig_match_results = []
                    for orig_sentence_split in original_italian_sentences_clean_split_by_sentences:
                        # Just taking the top matching sentence in the original Italian sentences
                        top_matching_sentence_in_line = process.extract(trans_split_sentence, orig_sentence_split, scorer=fuzz.ratio, limit=1)
                        orig_match_results.append(top_matching_sentence_in_line)

                    # Only take the 1 top matching sentence from orig_match_results
                    max_score = 0
                    max_matched_sentence = ""
                    for matched_sentence, similarity_score in orig_match_results:
                        if similarity_score > max_score:
                            max_score = similarity_score
                            max_matched_sentence = matched_sentence
                    match_results.append((max_matched_sentence, max_score))

                for result in match_results:
                    # CONSOLIDATE MATCH RESULTS and then output document with the maxim
                    
                print(f"\n--- Fuzzy Matching Results for Translation {i+1} ---")
                #print(f"English: '{english_text}'")
                print(f"Translation: '{italian_translation}'")
                print(f"\nTop {SHOW_TOP} matching original Italian sentences:")
                
                for j, (matched_sentence, similarity_score) in enumerate(match_results, 1):
                    with open(f"./data/{i}_{j}.json", 'w', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "similarity_score": similarity_score,
                            "english_text": english_text,
                            "italian_translation": italian_translation,
                            "matched_sentence": matched_sentence,
                            "in_sentence_split": ,
                            "matched_sentence_split": ,
                        }, ensure_ascii=False, indent=4))
                    print(f"{j:2d}. Score: {similarity_score:3d}% - '{matched_sentence[:100]}{'...' if len(matched_sentence) > 100 else ''}'")



        if FUZZY_MATCH_BY_LINE:
            for i, italian_translation in enumerate(translations):
                english_text = english_sentences[i]  # Get corresponding English text
                italian_translation_clean = clean_text(italian_translation)

                top_matching_documents = process.extract(italian_translation_clean, original_italian_sentences_clean, scorer=fuzz.ratio, limit=SHOW_TOP)
                
                print(f"\n--- Fuzzy Matching Results for Translation {i+1} ---")
                #print(f"English: '{english_text}'")
                print(f"Translation: '{italian_translation}'")
                print(f"\nTop {SHOW_TOP} matching original Italian sentences:")
                
                for j, (matched_sentence, similarity_score) in enumerate(top_matching_documents, 1):
                    with open(f"./data/{i}_{j}.json", 'w', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "similarity_score": similarity_score,
                            "english_text": english_text,
                            "italian_translation": italian_translation,
                            "matched_sentence": matched_sentence
                        }, ensure_ascii=False, indent=4))
                    print(f"{j:2d}. Score: {similarity_score:3d}% - '{matched_sentence[:100]}{'...' if len(matched_sentence) > 100 else ''}'")
                
                print("-" * 80)
