from typing import Any
import transformers
import torch

class Filler(torch.nn.Module):
    """
    A class that fills in masked tokens in a given text using a pre-trained language model.

    Args:
        lang (str): The language of the pre-trained language model. Currently supports 'swe' for Swedish and 'eng' for English.

    Raises:
        ValueError: If the specified language is not supported.

    Attributes:
        tokenizer (transformers.AutoTokenizer): The tokenizer for the pre-trained language model.
        model (transformers.ElectraForMaskedLM): The pre-trained language model.

    """
    def __init__(self, lang: str = "swe"):
        super().__init__()
        match lang.lower():
            case "swe": model_name = "KB/electra-small-swedish-cased-generator"
            case "eng": model_name = "google/electra-small-generator"
            case _:     raise ValueError(f"Language '{lang}' not supported.")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.ElectraForMaskedLM.from_pretrained(model_name)

    @torch.no_grad()
    def __call__(self, text: str) -> str:
        """
        Fills in masked tokens in the given text using the pre-trained language model.

        Args:
            text (str): The text to fill in masked tokens.

        Returns:
            str: The text with masked tokens filled in.

        """
        tokens = self.tokenizer(text, return_tensors="pt").input_ids
        output = self.model(tokens).logits.argmax(-1).squeeze()
        if len(output) != len(tokens.squeeze()):
            print(f"STRANGE!!!: {len(tokens.squeeze())} : {len(output)}")
        filled = self.tokenizer.decode(output[1:-1])
        return filled

class Whisperer(torch.nn.Module):
    class Whisperer:
        """
        A class that represents a whisperer that can fill in missing words in a sentence using the Chinese Whispers algorithm.

        Attributes:
            lang (str): The language used for filling in missing words. Defaults to 'swe'.
            filler (Filler): An instance of the Filler class used for filling in missing words.
        """
        def __init__(self, lang: str = "swe"):
            super().__init__()
            self.filler = Filler(lang)
    
    @torch.no_grad()
    def __call__(self, text: str, prob: float = 0.05) -> str:
        """
        Applies Chinese Whispers to the given text.

        Args:
            text (str): The text to apply Chinese Whispers to.
            prob (float, optional): The probability of a token being masked. Defaults to 0.01.

        Returns:
            str: The text with Chinese Whispers applied.

        """
        words = text.split(" ")
        mask = torch.rand(len(words)) < prob
        masked = ["[MASK]" if m else w for m, w in zip(mask, words)]
        masked_text = " ".join(masked)
        return self.filler(masked_text)
    
if __name__ == "__main__":
    import sys, time
    if "filler"in sys.argv:
        filler = Filler("eng" if "eng" in sys.argv else "swe")
        intro = "Enter text to fill in masked tokens Mask with ___: " if "eng" in sys.argv else \
                "Skriv in text för att fylla i maskerade ord. Maska med ___: "
        try:
            print(intro)
            while True:
                text = input().replace("___","[MASK]")
                print(filler(text))
        except KeyboardInterrupt:
            exit()
    else:
        whisperer = Whisperer("eng" if "eng" in sys.argv else "swe")
        intro = "Enter text to begin whisper chain: " if "eng" in sys.argv else \
                "Skriv in text för att starta viskleken: "
        text = input(intro)
        try:
            while True:
                print(text)
                text = whisperer(text)
                time.sleep(0.5)
        except KeyboardInterrupt:
            exit()