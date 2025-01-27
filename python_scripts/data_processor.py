# this file contains three classes:
# 1. GPT2Dataset: that wraps the text to Dataset object
# 2. GPT2DataLoader: that wraps the text to Dataloader object
# 3. GPT2ProcessorPipeLine: that build the dataset and dataloader and then return the embedding after passing the embedding weights.

import tiktoken, torch
from torch.utils.data import Dataset, DataLoader
from pydantic import FilePath

BPE_tokenizer = tiktoken.get_encoding('gpt2')

#-----------------------------------------------------------------------------#
class GPT2Dataset(Dataset):
    def __init__(self, txt_file: FilePath, tokenizer: tiktoken.core.Encoding = BPE_tokenizer,
                 max_length: int = 256, stride: int = 256):

        with open(txt_file,'r') as f:
            data_txt = f.read()
            data_ids =  tokenizer.encode(data_txt)

        self.vocab_size = tokenizer.n_vocab
        
        # This is to handle text size smaller than max-length ... 
        # -1 here because we depend on the smaller size which is the target not the input_ids
        max_length = len(data_ids)-1 if (len(data_ids) < max_length) else max_length

        self.input_ids = []
        self.target_ids = []

        for i in range(0,len(data_ids)-max_length,stride):
            self.input_ids.append(data_ids[i:i+max_length])
            self.target_ids.append(data_ids[i+1:i+max_length+1])

        self.context_size = max_length

    def __getitem__(self, index: int):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.target_ids[index])

    def __len__(self):
        return len(self.input_ids)
    
    @property
    def get_dataset(self):
        """ it is not used as much because we istanciate super().__init__"""
        return self
    
    @property 
    def get_vocab_and_context_size(self):
        return self.vocab_size, self.context_size

#-----------------------------------------------------------------------------#
class GPT2DataLoader(DataLoader):
    def __init__(self, dataset: Dataset, *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
    
    @property
    def get_dataloader(self):
        """ it is not used as much because we istanciate super().__init__"""
        return self


#-----------------------------------------------------------------------------#
class GPT2ProcessorPipeLine:
    def __init__(self, dataset: Dataset, dataloader: DataLoader, embedding_weights: torch.Tensor):
        self.ds = dataset
        self.dl = dataloader
        self.vocab_size, self.context_length = dataset.get_vocab_and_context_size

        self.embedding_layer = torch.nn.Embedding(num_embeddings = embedding_weights.shape[0],
                                                  embedding_dim = embedding_weights.shape[1])
        self.embedding_layer.weight = torch.nn.Parameter(embedding_weights)


    def process(self):
        # for simplicity, we assumed weights of positional embedding_layer are the same for token embedding one. 
        
        x_embedded = []
        y_embedded = []

        for i, (x,y) in enumerate(self.dl):
            print(f"Starting batch {i} >>>>> ")
            
            x_token_embedded = self.embedding_layer(x)
            y_token_embedded = self.embedding_layer(y)
            x_pos_embedded = self.embedding_layer(torch.arange(self.context_length).view(1,self.context_length))  # .view() is to add extra axis to match when summed with token_embedding.
            y_pos_embedded = self.embedding_layer(torch.arange(self.context_length).view(1,self.context_length))  

            x_embedded.append(x_token_embedded + x_pos_embedded)
            y_embedded.append(y_token_embedded + y_pos_embedded)
            
            print(f"Engind batch {i} <<<< ")
        return torch.stack(x_embedded, axis=0), torch.stack(y_embedded, axis=0)
            
#-----------------------------------------------------------------------------#
# Example usage:        

## loading dataset and dataloader objects
ds = GPT2Dataset(txt_file="the-verdict.txt", stride=256)
dl = GPT2DataLoader(ds, batch_size=2, num_workers=0, shuffle=False, drop_last=False)

## loading processor object
vocab_size, context_length = ds.get_vocab_and_context_size
embedding_weights = torch.nn.Embedding(vocab_size, 256).weight
processor = GPT2ProcessorPipeLine(ds, dl,embedding_weights)

## running processor to get the embedding of ds inputs and ds targets
ds_inputs_embedded, ds_targets_embedded = processor.process()
ds_inputs_embedded.shape
