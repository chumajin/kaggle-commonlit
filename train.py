import argparse
from pprint import pprint
import shutil

import time

from util import *


def initialization(fold):
    
    random_seed(cfg.seed)

    if cfg.fulltrain:
      p_train = train.copy()
      fold = 0
      cfg.evalstartepoch = cfg.stoptrain # multivalidationをここから始める

    else:
      p_train = train[train["fold"]!=fold].reset_index(drop=True)
    p_valid = train[train["fold"]==fold].reset_index(drop=True)

    valid_dataset = NLPDataSet(p_valid)

    tmp = []
    for a in range(len(valid_dataset)):
      tmp.append(len(valid_dataset[a]["ids"]))

    p_valid["length"] = tmp
    p_valid = p_valid.sort_values("length").reset_index(drop=True)

    train_dataset = NLPDataSet(p_train)
    valid_dataset = NLPDataSet(p_valid)

    train_dataloader = DataLoader(train_dataset,batch_size=cfg.train_batch,shuffle = True,num_workers=8,collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset,batch_size=cfg.valid_batch,shuffle = False,num_workers=8,collate_fn=collate)


    classifier = NLPModel(cfg.num_labels,cfg.model)
    if cfg.reinit:
        classifier = re_initializing_layer(classifier, classifier.config, cfg.reinit_n)
    classifier.to(device)

    grouped_optimizer_params = get_optimizer_grouped_parameters(classifier,
                                                                    cfg.layerwise_lr,
                                                                    cfg.layerwise_weight_decay,
                                                                    cfg.layerwise_lr_decay)


    optimizer = AdamW(grouped_optimizer_params,
                              lr = cfg.layerwise_lr,
                              eps = cfg.layerwise_adam_epsilon )

    num_train_steps = int(len(train_dataloader) * cfg.epochs)

    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    scaler = torch.cuda.amp.GradScaler()

    awp = None

    return p_train,p_valid,train_dataloader,valid_dataloader,classifier,optimizer,scheduler,scaler,awp





if __name__ == "__main__":
     

  # 0. parser setting

  parser = argparse.ArgumentParser()

  parser.add_argument("--modelno", type=str, default=2,help="makemodels")

  parser.add_argument('--train_fold', required=True, nargs="*", type=int,default=4, help='a list of training fold. 4 is full train')

  parser.add_argument("--savepath", type=str, default='.')
  parser.add_argument("--inputpath", type=str, default='.')
  parser.add_argument("--num_workers", type=int, default=8)

  parser.add_argument("--changelr", type=bool, default=False,help="if you want to change lr or not")
  parser.add_argument("--lr", type=float, default=5e-5,help="if you want to change lr, change this value")

  parser.add_argument("--changebatch", type=bool, default=False,help="if you want to change lr or not")
  parser.add_argument("--trainbatch", type=int, default=32,help="if you want to change trainbatch, change this value")
  parser.add_argument("--validbatch", type=int, default=32,help="if you want to change validbatch, change this value")

  parser.add_argument("--shutdown", type=bool, default=False,help="shutdown google colab after making the model(only colab)")

  opt = parser.parse_args()
  pprint(opt)

  # 1. change config

  import shutil

  script_path = os.path.abspath(__file__)
  script_directory = os.path.dirname(script_path)

  shutil.copy(f"{script_directory}/config/modelno{opt.modelno}.py",f"{script_directory}/cfg.py")
  from cfg import *

  if opt.changelr:
        cfg.lr = opt.lr
  if opt.changebatch:
        cfg.train_batch = opt.trainbatch
        cfg.valid_batch = opt.validbatch
  
  cfg.savepath = opt.savepath
  cfg.train_fold = opt.train_fold

  cfg.savevalid = cfg.savepath + "/savevalid"
  os.makedirs(cfg.savepath,exist_ok=True)
  os.makedirs(cfg.savevalid,exist_ok=True)

  tokenizer.save_pretrained(cfg.savepath)

  cfg.modelno = opt.modelno



  from metric import *
  from dataset_loader import *
  from pooling import *
  from model_utils import *
  from train_func import *
  from valid_func import *



  


  # 2. load samples

  sample = pd.read_csv(f"{opt.inputpath}/sample_submission.csv")
  test_prom = pd.read_csv(f"{opt.inputpath}/prompts_test.csv")
  train_prom = pd.read_csv(f"{opt.inputpath}/prompts_train.csv")
  train = pd.read_csv(f"{opt.inputpath}/summaries_train.csv")
  test = pd.read_csv(f"{opt.inputpath}/summaries_test.csv")

  train = train.join(train_prom.set_index("prompt_id"),on="prompt_id",how="left")
  test = test.join(test_prom.set_index("prompt_id"),on="prompt_id",how="left")
  train["fold"] = train["prompt_id"].map({'39c16e':0,'814d6b':1, 'ebad26':2, '3b9047':3})


  # 2.1 make 39 classification ref ) https://www.kaggle.com/code/alexandervc/commonlit-levenshtein-distances?scriptVersionId=141542492&cellId=23
  train["special"] = np.round(6.9*(0.63*train["content"]+(1-0.63)*train["wording"]))
  train["special"] = train["special"] + 10
  train["special"] = train["special"].astype("int")

  # 3. Dataset
  train_dataset = NLPDataSet(train)
  collate = Collate(tokenizer)

  # 4. main

  allvaliddf = []
  for fold in cfg.train_fold:

    print("")
    print(f"################  fold {fold} start ####################")
    print("")


    bestscore = np.inf
    beststep = 0

    cfg.fold = fold

    if fold == 4:
        cfg.fulltrain = True
        print("fulltrain start !!")

    p_train,p_valid,train_dataloader,valid_dataloader,model,optimizer,scheduler,scaler,awp = initialization(fold)

    allvalids = []
    allres2 = []


    if cfg.fulltrain_all:
        for epoch in range(4):
            allpreds,losses,score,model,_,_,_= training(
                train_dataloader,
                valid_dataloader,
                model,
                optimizer,
                scheduler,
                False,
                p_valid,
                fold,
                bestscore,
                epoch,
                cfg.fulltrain,
                awp
            )
        state = {
                            'state_dict': model.state_dict(),
                            #  'optimizer_dict': optimizer.state_dict(),
                            "bestscore":bestscore
                        }
        
        
        torch.save(state, os.path.join(cfg.savepath,f"modelno{opt.modelno}.pth"))
        print("fulltrain save")

        del state
        torch.cuda.empty_cache()
        gc.collect()

        break

    else:

        for epoch in range(4):

            if epoch >= cfg.evalstartepoch:
              cfg.evalstepswitch = True

            if cfg.evalstepswitch == False:

                allpreds,losses,score,model,_,_,_= training(
                    train_dataloader,
                    valid_dataloader,
                    model,
                    optimizer,
                    scheduler,

                    False,
                    p_valid,
                    fold,
                    bestscore,
                    epoch,
                    cfg.fulltrain,
                    awp
                )

                if cfg.fulltrain == False:


                    score,lossmean,p_valid2 = validating(valid_dataloader,p_valid,model,fold)


                    #### 2.1 Early stop ####

                    if bestscore > score:

                        print(f"Best score is {bestscore} → {score}. Saving model")
                        bestscore = score

                        state = {
                                    'state_dict': model.state_dict(),
                                    #  'optimizer_dict': optimizer.state_dict(),
                                    "bestscore":bestscore
                                }


                        torch.save(state, os.path.join(cfg.savepath,f"model{fold}_seed{cfg.seed}.pth"))
                        p_valid2.to_csv(f"{cfg.savepath}/valid{cfg.fold}_seed{cfg.seed}.csv",index=False)

                        del state
                        torch.cuda.empty_cache()
                        gc.collect()




                        beststep=0

                    else:
                        beststep +=1

                

            ## 3. multi validation ##

            else:
                allpreds,losses,score,model,stepbestscore,stepbestlossmean,p_valid2= training(
                    train_dataloader,
                    valid_dataloader,
                    model,
                    optimizer,
                    scheduler,
                    True,
                    p_valid,
                    fold,
                    bestscore,
                    epoch,
                    cfg.fulltrain,
                    awp
                )

                score = stepbestscore
                lossmean = stepbestlossmean

                if cfg.fulltrain == False:

                    if bestscore > stepbestscore:

                        print(f"Best score is {bestscore} → {stepbestscore}. ")

                        bestscore = stepbestscore



                        if cfg.loadmodel:

                            state = torch.load(os.path.join(cfg.savepath,f"model{fold}_seed{cfg.seed}.pth"))
                            model.load_state_dict(state["state_dict"])

                            del state

                            torch.cuda.empty_cache()
                            gc.collect()

                        beststep=0

                    else:
                        beststep += 1

                

            if cfg.fulltrain == False:
                print(f"epoch : {epoch}, train loss : {losses}, valid loss : {lossmean}, score : {score}, bestscore : {bestscore}")
                allvalids.append(p_valid2)

            p_valid2 = pd.read_csv(f"{cfg.savepath}/valid{fold}_seed{cfg.seed}.csv")
            allvaliddf.append(p_valid2)

            tmpdf = pd.DataFrame({"test":[1]})
            tmpdf.to_csv(f"{cfg.savepath}/end{fold}_score{bestscore:.4f}.csv",index=False)

  # 5. makeoof

  filepath = [os.path.join(cfg.savepath,s) for s in os.listdir(cfg.savepath) if "end" in s]
  labels = ["content","wording"]
  usecols = ["content_pred","wording_pred"]

  if len(filepath) == 4:
    for fold,path in enumerate(filepath):
        tmpdf = pd.read_csv(f"{cfg.savepath}/valid{fold}_seed{cfg.seed}.csv")
        allvaliddf.append(tmpdf)


    allvaliddf = pd.concat(allvaliddf,ignore_index=True)
    score = get_score(allvaliddf[labels].values,allvaliddf[usecols].values)
    print(score)
    allvaliddf.to_csv(f"{cfg.savepath}/allvaliddf_exp{EXP}_{score:.05}.csv",index=False)

  else:
    if opt.shutdown:
        from google.colab import runtime    
        runtime.unassign()




















