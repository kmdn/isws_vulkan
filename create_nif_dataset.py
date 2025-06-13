from pynif import NIFCollection
import csv


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
                sentences.append((row[0], row[1]))
    return sentences


def get_entities(filepath):
    """ Reads entities from CSV file (entities.csv).
    Args:
        filepath (str): Path to the file containing entities.
    Returns:
        list: List of entities.
    """
    entities = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Ignore header
        next(reader)
        for row in reader:
            if row:
                identifier = row[0]
                start_offset = int(row[1])
                end_offset = int(row[2])
                mention = row[3]
                ta_class_ref = row[4]
                ta_ident_ref = row[5]
                entities.append((identifier, start_offset, end_offset, mention, ta_class_ref, ta_ident_ref))
    return entities


def get_type_uri(type_name):
    """ Converts a NER type to a Wikidata URI.
    E.g. 
    Person -> https://www.wikidata.org/wiki/Q5
    Location -> https://www.wikidata.org/wiki/Q2221906
    'PER', 'MISC', 'ORG', 'LOC'
    Organisation -> https://www.wikidata.org/wiki/Q43229
    Miscellaneous -> https://www.wikidata.org/wiki/Q35120
    """
    type_mapping = {
        'PER': 'Q5',  # Person
        'LOC': 'Q2221906',  # Location
        'ORG': 'Q43229',  # Organisation
        'MISC': 'Q35120'  # Miscellaneous
    }
    return f"https://www.wikidata.org/wiki/{type_mapping.get(type_name, 'Q35120')}"

base_uri = "https://2025.semanticwebschool.org/"

gt_sentences = get_english_sentences('./data/sentences.csv')

gt_entities = get_entities('./data/entities.csv')

# Group entities by their identifier
dict_entities_by_id = {
    entity[0]: {
        'start_offset': entity[1],
        'end_offset': entity[2],
        'mention': entity[3],
        'type': entity[4],
        'uri': entity[5]
    } for entity in gt_entities
}

collection = NIFCollection(uri=base_uri)
for idx, document in gt_sentences:
    context = collection.add_context(
        uri=base_uri + str(idx),
        mention=document)

    if dict_entities_by_id.get(idx, None):
        entity = dict_entities_by_id[idx]
        context.add_phrase(
            beginIndex=int(entity['start_offset']),
            endIndex=int(entity['end_offset']),
            taClassRef=[get_type_uri(entity['type'])],
            #score=1,
            annotator='ISE-FIZKarlsruhe',
            taIdentRef=f"https://www.wikidata.org/wiki/{entity['uri']}",
            #taMsClassRef=entity['type'][0] if entity['type'] else None
            )

generated_nif = collection.dumps(format='turtle')
#print(generated_nif[:5000])
# Dump generated NIF to a file
with open('vasari_en.nif', 'w', encoding='utf-8') as file:
    file.write(generated_nif)