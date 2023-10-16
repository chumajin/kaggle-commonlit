
EXP = 4

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

    seed = 277
    maxlen = 868
    model = "microsoft/deberta-v3-large"
    #pooling
    pooling = 'none' # mean, max, min, attention, weightedlayer, cls
    layerwise_lr = 5e-5 

    stoptrain = 3
    stopvalidcount = 6 # stop fulltrain at this point

    fulltrain_all = True # if allfull train

    textlength = False

    arcface = True


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

    ## 5. arcface
    arcface = True
    s =  13
    m = 0.0001
    margin = False
    eps = 0.0

    emb_dim = 1024


cfg = CFG()

tokenizer = AutoTokenizer.from_pretrained(cfg.model)
tokenizer.save_pretrained(cfg.savepath)

## Dataset

class NLPDataSet(Dataset):
    
    def __init__(self,df):

        self.df = df.copy()

        self.prompt = self.df["prompt_text"].map(cleantext2).values
        self.text = self.df["text"].map(cleantext2).values + " : " +self.df["prompt_title"].map(cleantext2).values  + " : " + self.df["prompt_question"].map(cleantext2).values

        self.special = self.df["special"].values


    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):


        tokens = tokenizer.encode_plus(self.text[idx],
                                       self.prompt[idx],
                                          return_tensors = None,
                                          add_special_tokens = True,
                                          max_length = cfg.maxlen,
                                      #    pad_to_max_length = True, # これを入れるかは微妙。collate使うならいらない気がする poolingのとき、collateあり。
                                      #    truncation = True
                                          )



        ids = torch.tensor(tokens['input_ids'], dtype=torch.long)
        mask = torch.tensor(tokens['attention_mask'], dtype=torch.long)
        target = self.df[label].iloc[idx].values
        target2 = self.special[idx]
        token_type_ids = torch.tensor(tokens['token_type_ids'], dtype=torch.long)


        return {
                  'ids': ids,
                  'mask': mask,
                  'token_type_ids': token_type_ids,
                  'targets': target,
                  "targets2":target2

              }

import math

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0,
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


def linear_combination(x, y, epsilon):
    return (1 - epsilon) * x + epsilon * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.05, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(nll, loss/n, self.epsilon)

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

        self.bn = nn.BatchNorm1d(self.config.hidden_size)
        self.dropout = nn.Dropout(0)
        self.fc2 = nn.Linear(self.config.hidden_size, cfg.emb_dim)
        self.bn2 = nn.BatchNorm1d(cfg.emb_dim)

        self.head = ArcMarginProduct(cfg.emb_dim,
                                   38,
                                   s=cfg.s,
                                   m=cfg.m,
                                   easy_margin=cfg.margin,
                                   ls_eps=cfg.eps)
        self._init_weights(self.fc2)




    def forward(self, ids, mask, token_type_ids=None, targets=None,targets2=None):


        output = self.model(ids, mask, token_type_ids)
        output = output[0][:,0,:]

        features = self.bn(output)
        features = self.dropout(features)
        features = self.fc2(features)
        features = self.bn2(features)
        output2 = self.head(features,targets2)
        output = self.fc(output)


        loss = 0
        metrics = 0

        if targets is not None:

          loss = self.loss(output, targets)
          loss2 = self.loss2(output2,targets2)

          loss = 0.7*loss + 0.3 * loss2
#          metrics = self.monitor_metrics(output, targets)

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

    def loss2(self, outputs, targets):

        loss_fct = LabelSmoothingCrossEntropy()
        loss = loss_fct(outputs.squeeze(-1), targets.squeeze(-1))

        return loss

cfg = CFG()