
EXP = 2

compname = "commonlit-evaluate-student-summaries"

import os
from util import *
from pooling import *

## config ##

class CFG():

# ============== 0. basic =============

    input_dir = f"./{compname}"

    savepath =  f"./Commonlit-modelno{EXP}"
    savevalid = savepath + "/savevalid"

    sessionend = True
    traintype = "multiregression" # binary / regression / classification / multiregression


# ============== 1. change parts =============

    seed = 179
    maxlen = 1024
    model = "microsoft/deberta-v3-large"
    #pooling
    pooling = 'attention' # mean, max, min, attention, weightedlayer, cls
    layerwise_lr = 7.5e-5 

    stoptrain = 3
    stopvalidcount = 5 # stop fulltrain at this point

    fulltrain_all = False # if allfull train


# ============== 2.Data  =============

    train_batch = 32
    valid_batch = 32

    train_fold = [0,1,2,3] # you can change here to train multi folds. [0,1,2,3]. fulltrain : 4

    
# ============== 3. Model & Training (fix) =============

    
    evalstepswitch = False # For multi validation switch
    evalstartepoch = 1 # even if the switch is False, after this epoch, switch will be on.
    evalstepnum = 5 # 5 times validation per epoch
    loadmodel = False # fix

    num_labels = 2

    accumulation_steps = 1
    max_grad_norm = 1000

    epochs=4
    scheduler='cosine' # ['linear', 'cosine']

    grad_check = True # gradient check point

    layer_start = 4 
    #init_weight
    init_weight = 'normal' # normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal
    #re-init
    reinit = True
    reinit_n = 1

    #Layer-Wise Learning Rate Decay
    llrd = True # switch
    layerwise_lr_decay = 0.9 
    layerwise_weight_decay = 0.01 
    layerwise_adam_epsilon = 1e-6 
    layerwise_use_bertadam = False


    collate = True
    fulltrain = False


cfg = CFG()
os.makedirs(cfg.savepath, exist_ok=True)
os.makedirs(cfg.savevalid,exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(cfg.model)
tokenizer.save_pretrained(cfg.savepath)

## Dataset

class NLPDataSet(Dataset):
    
    def __init__(self,df):

        self.df = df.copy()

        self.prompt = self.df["prompt_title"].values  + " " + tokenizer.sep_token + " " + self.df["prompt_question"].values + " " + tokenizer.sep_token + " " + self.df["prompt_text"].values
        self.text = "Evaluating the summarized text and calculating content and wording score : " + self.df["text"].values

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):


        tokens = tokenizer.encode_plus(self.text[idx],
                                       self.prompt[idx],
                                          return_tensors = None,
                                          add_special_tokens = True,
                                          max_length = cfg.maxlen,
                                          )

        ids = torch.tensor(tokens['input_ids'], dtype=torch.long)
        mask = torch.tensor(tokens['attention_mask'], dtype=torch.long)
        target = self.df[label].iloc[idx].values
        token_type_ids = torch.tensor(tokens['token_type_ids'], dtype=torch.long)

        return {
                  'ids': ids,
                  'mask': mask,
                  'token_type_ids': token_type_ids,
                  'targets': target

              }

class Collate:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]


        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        if "targets" in batch[0]:
            output["targets"] = [sample["targets"] for sample in batch]

            if cfg.traintype=="classification":
              output["targets"] =torch.tensor( output["targets"], dtype=torch.long)

            else:
              output["targets"] =torch.tensor( output["targets"], dtype=torch.float)

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


        ### token type ids ###

        output["token_type_ids"] = [sample["token_type_ids"] for sample in batch]
        if tokenizer.padding_side == "right":
            output["token_type_ids"] =[list(np.array(s)) + (batch_max - len(s)) * [0] for s in output["token_type_ids"]]
        else:
            output["token_type_ids"] =[(batch_max - len(s)) * [0] + np.array(s) for s in output["token_type_ids"]]
        output["token_type_ids"] = torch.tensor(output["token_type_ids"], dtype=torch.long)


        return output
    
class NLPModel(nn.Module):
    def __init__(self,num_labels,model_name):

        super().__init__()
        self.num_labels = num_labels

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.save_pretrained(cfg.savepath)

        ## config set ##

        layer_norm_eps: float = 1e-7
        self.config.layer_norm_eps = layer_norm_eps
        self.config.output_hidden_states = True

        self.config.hidden_dropout = 0.007
        self.config.hidden_dropout_prob = 0.007
        self.config.attention_dropout = 0.007
        self.config.attention_probs_dropout_prob = 0.007


        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.model.save_pretrained(cfg.savepath)

        # gradient point

        if cfg.grad_check:
          self.model.gradient_checkpointing_enable()


        ## pooling ##

        if cfg.pooling == 'mean':
            self.pool = MeanPooling()
        elif cfg.pooling == 'max':
            self.pool = MaxPooling()
        elif cfg.pooling == 'min':
            self.pool = MinPooling()
        elif cfg.pooling == 'attention':
            self.pool = AttentionPooling(self.config.hidden_size)
        elif cfg.pooling == 'weightedlayer':
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, layer_start = cfg.layer_start, layer_weights = None)

        self.fc = nn.Linear(self.config.hidden_size, 2)
        self._init_weights(self.fc)




    def forward(self, ids, mask, token_type_ids=None, targets=None):

        output = self.model(ids, mask, token_type_ids)

        output = output[0][:,:12,:]
        output = self.pool(output,mask[:,:12])
        output = self.fc(output)


        loss = 0
        metrics = 0

        if targets is not None:

          loss = self.loss(output, targets)

        output = output.detach().cpu().numpy()

        return output,loss, metrics



    #### https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train#Model ###

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if cfg.init_weight == 'normal':
                module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            elif cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if cfg.init_weight == 'normal':
                module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            elif cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

  #### loss function #####################

    def loss(self, outputs, targets):

        loss_fct = nn.SmoothL1Loss(reduction='mean')
        loss = loss_fct(outputs.squeeze(-1), targets.squeeze(-1))

        return loss