# %%
import re

REGEX_PUNCTUATIONS: str = r'([,.:;?_!"()\']|--|\s)'
REGEX_PUNCTUATIONS_DECODE: str = r'\s+([,.?!"()\'])'
UNKNOWN_TOKEN: str = "<|unk|>"
END_OF_TEXT_TOKEN: str = "<|endoftext|>"

def preprocess_text(text):
    tokens = re.split(REGEX_PUNCTUATIONS, text)
    tokens = [item for item in tokens if item.strip()]
    tokens.extend([END_OF_TEXT_TOKEN, UNKNOWN_TOKEN])
    return tokens

def build_vocab(preprocessed_text):
    all_words = sorted(set(preprocessed_text))
    return {token:integer for integer, token in enumerate(all_words)}

class SimpleTokenizerV1:
    regex_punctuations: str = REGEX_PUNCTUATIONS
    regex_punctuations_decode: str = REGEX_PUNCTUATIONS_DECODE

    def __init__(self, vocab: dict[str,int]) -> None:
        """
        Initialize Tokenizer
        """
        self.str_to_int: dict[str, int] = vocab
        self.int_to_str: dict[int, str] = {i:word for word, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        Encode a sentence or paragraph or any text into a sequence of token ids.
        """
        preprocessed_text = re.split(self.regex_punctuations, text)
        token_ids = [self.str_to_int[item] for item in preprocessed_text if item.strip()]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a sequence of tokens to text.
        """
        text = " ".join([self.int_to_str[token_id] for token_id in token_ids])
        text = re.sub(self.regex_punctuations_decode, r'\1', text)
        return text

# %%

class SimpleTokenizerV2:
    regex_punctuations: str = REGEX_PUNCTUATIONS
    regex_punctuations_decode: str = REGEX_PUNCTUATIONS_DECODE

    def __init__(self, vocab: dict[str,int]) -> None:
        """
        Initialize Tokenizer
        """
        self.str_to_int: dict[str, int] = vocab
        self.int_to_str: dict[int, str] = {i:word for word, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        Encode a sentence or paragraph or any text into a sequence of token ids.
        """
        preprocessed_text = re.split(self.regex_punctuations, text)
        preprocessed_text = [item for item in preprocessed_text if item.strip()]
        preprocessed_text = [item if item in self.str_to_int else UNKNOWN_TOKEN for item in preprocessed_text]
        token_ids = [self.str_to_int[item] for item in preprocessed_text]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a sequence of tokens to text.
        """
        text = " ".join([self.int_to_str[token_id] for token_id in token_ids])
        text = re.sub(self.regex_punctuations_decode, r'\1', text)
        return text
# %%
