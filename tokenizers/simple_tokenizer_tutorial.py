# %% Download the text file
import urllib.request
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as file:
    raw_text = file.read()

print(f"Total Number of characters in the text: {len(raw_text)}")
print(f"First 100 characters of the text:\n{raw_text[:100]}")

# %% Regex split the words in the text
import re
test_text = "Hello, World! This is a test."
result = re.split(r'(\s)', test_text)
# This regex splits the text by whitespace
print(result)
# However, this still has some of the punctuation attached to the words.

# Update the regex to split by whitespace and punctuation
result = re.split(r'([,.]|\s)', test_text)
print(result)

# Finally, remove the white spaces
result = [item for item in result if item.strip()]
print(result)

# Please note that there are several applications where whitespaces might be necessary to be included in the tokenizer.
# %% Create a tokenizer that includes all kinds of punctuation

regex_punctuations = r'([,.:;?_!"()\']|--|\s)'
test_text = "Hello, World! This is a (test)--isn't it great?"
result = re.split(regex_punctuations, test_text)
result = [item for item in result if item.strip()]
print(result)

# %% create a simple preprocessor function

def preprocess_text(text):
    tokens = re.split(regex_punctuations, text)
    tokens = [item for item in tokens if item.strip()]
    return tokens

preprocessed_text = preprocess_text(raw_text)
print(f"First 100 tokens:\n{preprocessed_text[:100]}")
# %% Create a vocabulsary from the preprocessed text

all_tokens = sorted(set(preprocessed_text))
print(f"First 100 unique tokens:\n{all_tokens[:100]}")
vocab_size = len(all_tokens)

word_to_token_id = {}
token_id_to_word = {}
for id, word in enumerate(all_tokens):
    word_to_token_id[word] = id
    token_id_to_word[id] = word
# %%
