from transformers import pipeline
from typing_extensions import Self
from typing import Literal
import warnings, re

class AIMasker:
    """
    A class for masking sensitive information in text using Named Entity Recognition (NER) and regular expressions.

    Attributes:
    -----------
    model_name : str
        The name of the pre-trained NER model to use for masking. Defaults to "KBLab/bert-base-swedish-lowermix-reallysimple-ner" for Swedish language, and "dslim/bert-base-NER" for other languages.
    NER : transformers.pipelines.Pipeline
        The NER pipeline object from the transformers library.
    regexes : dict
        A dictionary containing regular expressions for matching sensitive information in text. The keys should be all caps.
    entity_word_occurences : list or None
        A list of words to ignore during masking. If None, all words will be masked.
    entity_words : dict or None
        A dictionary containing the original words for each entity found in the text. If None, no entities have been found in the text.
    text : str or None
        The text to be masked. If None, no text has been provided for masking.
    """

    def __init__(self, lang: str = "swe"):
        self.model_name = "KBLab/bert-base-swedish-lowermix-reallysimple-ner" if lang == "swe" else "dslim/bert-base-NER"
        self.NER = pipeline("ner", model=self.model_name, tokenizer=self.model_name)
        self.regexes = {
            "PHN": r"[+]?[0-9]{2}[-\s]?[0-9]{1,2}[-\s]?[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{2}", # Match phone numbers
            "IDN": r"[12]?[09]?[0-9]{6}[-\s]?[0-9]{4}", # Match personal identitys numbers
            "TME": r"[012]{2}:[0-9]{2}", # Match times
        } # Note that all regex names should be all caps
        self.entity_word_occurences = None
        self.entity_words = None
        self.text = None

    def _ner_entities(self, text: str) -> list[dict[str: any]]:
        """
        Returns the entities in the text using the pre-trained language model.

        Args:
            text (str): The text to read.

        Returns:
            list: A list of entities.

        """
        token_entities = self.NER(text)
        word_entities = []
        # The NER might split words into tokens, so we need to merge them
        for token in token_entities:
            if token['word'].startswith('##'):
                word_entities[-1]['word'] += token['word'][2:]
            else:
                word_entities += [ token ]
        return word_entities

    def _reg_entities(self, text: str) -> list[dict[str: any]]:
        entities = []
        for entity, regex in self.regexes.items():
            for match in re.finditer(regex, text):
                entities.append({
                    "entity": entity,
                    "word": match.group(),
                    "start": match.start(),
                })
        return entities
    
    def read(self, text: str) -> Self:
        """
        Reads a text and returns a list of entities. Should be called before masking or requesting entities.

        Args:
            text (str): The text to read.

        Returns:
            list: A list of entities.

        """
        entity_words = {}
        entity_word_occurences = {}
        entities = self._ner_entities(text) + self._reg_entities(text)
        for item in entities:
            # Collect all words for each entity
            if item["entity"] not in entity_words:
                entity_words[item["entity"]] = []
            if item['word'] not in entity_words[item["entity"]]:
                entity_words[item['entity']].append(item['word'])

            # Collect the occurences for each word, split over entities
            if item["entity"] not in entity_word_occurences:
                entity_word_occurences[item["entity"]] = {}
            if item["word"] not in entity_word_occurences[item["entity"]]:
                entity_word_occurences[item["entity"]][item["word"]] = []
            entity_word_occurences[item["entity"]][item["word"]].append(item["start"])

        self.text = text
        self.entity_word_occurences = entity_word_occurences
        self.entity_words = entity_words
        return self

    def entities(self) -> dict[str: list[str]]:
        """
        Returns the entities in the text. Should be called after read().

        Returns:
            list[str]: A list of entities.

        """
        if self.entity_words is None:
            raise ValueError("Must call read() before entities()!")
        return self.entity_words

    def mask(self, to_mask: dict[str: list[str]], mode: Literal["mask","anon"] = "mask") -> str:
        """
        Masks the given entities in the text.

        Args:
            to_mask (dict[str: list[str]]): A dictionary [entity -> [words]] to mask.

        Returns:
            str: The text with the given entities masked.

        """
        if self.entity_word_occurences is None or self.text is None:
            raise ValueError("Must call read() before mask()!")
        masked_text = self.text
        offsets = {}
        for entity, words in to_mask.items():
            entity = entity.upper()
            for word in words:
                if entity not in self.entity_word_occurences:
                    warnings.warn(f"'{entity}' not found in: '{self.entity_word_occurences.keys()}'.")
                    continue # Can't mask a word that doesn't exist
                if  word not in self.entity_word_occurences[entity]:
                    warnings.warn(f"Word '{word}' not found in entity '{entity}'.")
                    continue # Can't mask a word that doesn't exist
                for occurence in self.entity_word_occurences[entity][word]:
                    match mode.lower():
                        case "mask": mask = "[MASK]"
                        case "anon": mask = entity + str(self.entity_words[entity].index(word)+1)
                        case _: raise ValueError(f"Mode '{mode}' not supported.")
                    offset = sum([o for p, o in offsets.items() if p < occurence])
                    start = occurence+offset
                    end = start+len(word)
                    offsets[occurence] = len(mask) - len(word)
                    masked_text = masked_text[:start] + mask + masked_text[end:]
        return masked_text
    
if __name__ == "__main__":
    import sys
    am = AIMasker()
    intro = "Enter text to mask:" if "eng" in sys.argv else "Skriv in en text att maska:"
    print(intro)
    text = input()
    am.read(text)
    entity_text = "The text contains the following entities:" if "eng" in sys.argv else "Texten innehåller följande entiteter:"
    for entity, words in am.entities().items():
        print(f"   {entity}: {', '.join(words)}")
    mask_text = "Enter entities to mask (i.e. PRS Erik):" if "eng" in sys.argv else "Skriv in entiteter att maska (ex. PRS Erik):"
    print(mask_text)
    to_mask = {}
    while(masks := input()):
        entity, word = masks.split(" ")
        to_mask[entity] = to_mask.get(entity, []) + [word]
    masked_text = am.mask(to_mask, mode="mask" if not "anon" in sys.argv else "anon")
    outro = "Masked text:" if "eng" in sys.argv else "Maskad text:"
    print(outro)
    print(masked_text)