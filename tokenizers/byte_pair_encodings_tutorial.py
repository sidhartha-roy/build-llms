# %%
from importlib.metadata import version
import tiktoken


print("tiktoken version:", version("tiktoken"))
# %%

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)


"""
Note: The BPE tokenzier deals with unknown words by dealing with them algorithmically. For example, the
BPE tokeinzer algorithm starts with all the characters and then builds up by picking up sub words such as de, am, pi based on a frequency cut off.
Then it keeps adding larger and larger sub words and builds up the vocabulary. Therefore, BPE can encode any word and does not have any unknown tokens.

"""
# %%

