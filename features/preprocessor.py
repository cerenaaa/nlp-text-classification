"""Text preprocessing: cleaning, normalization, tokenization."""
import re
import string


class TextPreprocessor:
    def __init__(self, lowercase: bool = True, remove_punct: bool = True,
                 remove_numbers: bool = False, max_len: int = 512):
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_numbers = remove_numbers
        self.max_len = max_len

    def clean(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        if self.remove_punct:
            text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text[:self.max_len]

    def process_batch(self, texts: list[str]) -> list[str]:
        return [self.clean(t) for t in texts]
