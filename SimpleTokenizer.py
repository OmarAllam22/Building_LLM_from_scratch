from pydantic import validate_call, FilePath
import re

class SimpleTokenizer:
    
    @validate_call
    def __init__(self, file_path: FilePath) -> None:
        with open(file_path, "r") as f:
            txt = f.read()
            txt = re.findall(r"\w+|[^\w\s]", txt)
            txt += ["<|unk|>", "<|endoftext|>"]
            vocabs_dict = dict(enumerate(set(txt)))

        self.int2str = vocabs_dict
        self.str2int = {v: i for i, v in vocabs_dict.items()}
    
    @validate_call
    def encode(self, txt: str) -> list[int]:
        return [self.str2int.get(token) for token in txt.split() + ["<|endoftext|>"]]
    
    @validate_call
    def decode(self, token_ids: list[int]) -> str:
        return " ".join([self.int2str.get(id, "<|unk|>") for id in token_ids])


#example usage
o = SimpleTokenizer(file_path="the-verdict.txt")
ids = o.encode("How are you there")
o.decode(ids)
