
EXP = 3

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

    seed = 237
    maxlen = 850
    model = "microsoft/deberta-v3-large"
    #pooling
    pooling = 'none' # mean, max, min, attention, weightedlayer, cls
    layerwise_lr = 7.5e-5 

    stoptrain = 3
    stopvalidcount = 2 # stop fulltrain at this point

    fulltrain_all = False # if allfull train

    textlength = True

    arcface = False


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

    num_cycles=0.5
    num_warmup_steps=0
    warmupratio = 0
    scheduler='cosine' # ['linear', 'cosine']
    epochs = 4



    collate = True
    fulltrain = False


cfg = CFG()

tokenizer = AutoTokenizer.from_pretrained(cfg.model)

## Dataset

class NLPDataSet(Dataset):
    
    def __init__(self,df):

        self.df = df.copy()

        self.text = self.df["text"].values + " : " + self.df["prompt_question"].values + " : " + self.df["prompt_title"].values + " : "  + self.df["prompt_text"].values
        self.text2 = self.df["text"].values

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):


        tokens = tokenizer.encode_plus(self.text[idx],
                                          return_tensors = None,
                                          add_special_tokens = True,
                                          max_length = cfg.maxlen,
                                      #    pad_to_max_length = True, # これを入れるかは微妙。collate使うならいらない気がする poolingのとき、collateあり。
                                      #    truncation = True
                                          )


        tokens2 = tokenizer.encode_plus(self.text2[idx],
                                          return_tensors = None,
                                          add_special_tokens = True,
                                          max_length = cfg.maxlen,
                                      #    pad_to_max_length = True, # これを入れるかは微妙。collate使うならいらない気がする poolingのとき、collateあり。
                                      #    truncation = True
                                          )

        textlength = len(tokens2["input_ids"])

        ids = torch.tensor(tokens['input_ids'], dtype=torch.long)
        mask = torch.tensor(tokens['attention_mask'], dtype=torch.long)
        target = self.df[label].iloc[idx].values
        token_type_ids = torch.tensor(tokens['token_type_ids'], dtype=torch.long)


        return {
                  'ids': ids,
                  'mask': mask,
                  'token_type_ids': token_type_ids,
                  'targets': target,
                  "textlength":textlength

              }



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
            self.pool = AttentionPooling(self.config.hidden_size*2)
        elif cfg.pooling == 'weightedlayer':
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, layer_start = cfg.layer_start, layer_weights = None)

        self.fc = nn.Linear(self.config.hidden_size, 2)
        self._init_weights(self.fc)


    def forward(self, ids, mask, token_type_ids,textlength, targets=None):

        output = self.model(ids, mask, token_type_ids)
        bsize = len(ids)

        pooled_outputs = []
        hidden_states = output[0]
        for i in range(bsize):
            pooled_output = hidden_states[i,:textlength[i]].mean(dim=0)
            pooled_outputs.append(pooled_output)
        output = torch.stack(pooled_outputs)


        output = self.fc(output)

        loss = 0
        metrics = 0

        if targets is not None:

          loss = self.loss(output, targets)

        output = output.detach().cpu().numpy()


        return output, loss, metrics



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
    
cfg = CFG()

