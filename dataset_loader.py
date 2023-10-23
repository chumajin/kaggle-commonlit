from util import *
from cfg import *

class Collate:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]

        if cfg.textlength:
            output["textlength"] = [sample["textlength"] for sample in batch]


        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        output["targets"] = [sample["targets"] for sample in batch]
        output["targets"] =torch.tensor( output["targets"], dtype=torch.float)

        if cfg.arcface:
            output["targets2"] = [sample["targets2"] for sample in batch]
            output["targets2"] =torch.tensor( output["targets2"], dtype=torch.long)


        # add padding
        if tokenizer.padding_side == "right":
            output["ids"] = [list(np.array(s)) + (batch_max - len(s)) * [tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [list(np.array(s)) + (batch_max - len(s)) * [0] for s in output["mask"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [tokenizer.pad_token_id] + np.array(s) for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + np.array(s) for s in output["mask"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)

        output["token_type_ids"] = [sample["token_type_ids"] for sample in batch]
        if tokenizer.padding_side == "right":
            output["token_type_ids"] =[list(np.array(s)) + (batch_max - len(s)) * [0] for s in output["token_type_ids"]]
        else:
            output["token_type_ids"] =[(batch_max - len(s)) * [0] + np.array(s) for s in output["token_type_ids"]]
        output["token_type_ids"] = torch.tensor(output["token_type_ids"], dtype=torch.long)


        return output
