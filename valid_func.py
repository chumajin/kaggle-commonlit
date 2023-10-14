from util import *
from metric import *
from cfg import *

def validating(valid_dataloader,p_valid,model,fold):
    
    model.eval()
    torch.backends.cudnn.benchmark = True

    allpreds = []
    alltargets = []

    score = 0

    p_valid33 = p_valid.copy()

    num = 0
    num2 = 0

    for step,a in tqdm(enumerate(valid_dataloader)):

            losses = []

            with torch.no_grad():

                ids = a["ids"].to(device,non_blocking=True) # non_blocking=TrueでPinned MemoryからGPUに転送中もCPUが動作できるらしい。
                mask = a["mask"].to(device,non_blocking=True)
                targets = a["targets"].to(device,non_blocking=True)

                tokentype = a["token_type_ids"].to(device,non_blocking=True)
                logits, loss, metric = model(ids,mask, token_type_ids=tokentype,targets=targets)

                losses.append(loss.mean().item())

                targets = targets.detach().cpu().numpy()

                allpreds.append(logits)
                alltargets.append(targets)

    allpreds = np.concatenate(allpreds)
    alltargets = np.concatenate(alltargets)


    #################
    ### make oof ####
    #################

    preddf = pd.DataFrame(allpreds)
    preddf.columns = ["content_pred","wording_pred"]
    cols = preddf.columns.to_list()

    p_valid33 = pd.concat([p_valid33,preddf],axis=1)

    score = get_score(p_valid33[label].values,p_valid33[cols].values)
    print(f"fold {fold} score is {score}")

    lossmean = np.mean(losses)

    del logits,loss,ids,mask,targets,metric
    torch.cuda.empty_cache()
    gc.collect()


    return score,lossmean,p_valid33