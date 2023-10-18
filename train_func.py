from util import *
from metric import *
from cfg import *
from valid_func import *
from dataset_loader import *

def training(
    train_dataloader,
    valid_dataloader,
    model,
    optimizer,
    scheduler,
    evalstepswitch,
    p_valid,
    fold,
    bestscore,
    epoch,
    fgm = None
):

    model.train()

    cols = ["content_pred","wording_pred"]

    torch.backends.cudnn.benchmark = True

    allpreds = []
    alltargets = []
    allpreds_index = []

    #bestscore = 0
    bestlossmean = 0

    accumloss = 0

    valcount = 0

    t=time.time()

    steps_per_epoch = int(cfg.train_batch / len(train_dataloader)) + 1

    valstep = int(len(train_dataloader) / (cfg.evalstepnum + 1))



    global_step = 0


    for step,a in enumerate(tqdm(train_dataloader)):

        losses = []



        with torch.cuda.amp.autocast():


            ids = a["ids"].to(device,non_blocking=True)
            mask = a["mask"].to(device,non_blocking=True)
            targets = a["targets"].to(device,non_blocking=True)
            tokentype = a["token_type_ids"].to(device,non_blocking=True)

            if cfg.textlength:
                  textlength = a["textlength"]
                  logits, loss, metric = model(ids,mask, tokentype,textlength,targets=targets)
            elif cfg.arcface:
                  targets2 = a["targets2"].to(device,non_blocking=True)
                  logits, loss, metric = model(ids,mask, token_type_ids=tokentype,targets=targets,targets2=targets2)
            else:
                  logits, loss, metric = model(ids,mask, token_type_ids=tokentype,targets=targets)

            losses.append(loss.mean().item())

            allpreds.append(logits)
            alltargets.append(targets.detach().cpu().numpy())

        loss = loss.mean()/cfg.accumulation_steps
        scaler.scale(loss).backward()

        if cfg.max_grad_norm >0:
          grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm) # gradnorm追加


        #### accumulation step ###

        if (step+1) % cfg.accumulation_steps ==0:

            scaler.step(optimizer) # オプティマイザーの更新
            scaler.update() # スケーラーの更新
            optimizer.zero_grad()
            scheduler.step()

        if (step+1) % 100 ==0:

          if cfg.max_grad_norm >0:
            print ("Step [{}/{}] Loss: {:.3f} Time: {:.1f} Metric: {:.3f} lr : {} Grad : {}".format(step+1, steps_per_epoch, loss.item(), time.time()-t,0,scheduler.get_lr()[0],grad_norm))
          else:
            print ("Step [{}/{}] Loss: {:.3f} Time: {:.1f} Metric: {:.3f} lr : {}".format(step+1, steps_per_epoch, loss.item(), time.time()-t,0,scheduler.get_lr()[0]))

          torch.cuda.empty_cache()
          gc.collect()


      ### multi validation ####


        if evalstepswitch:

            if (step+1) % valstep == 0 :

                score,lossmean,p_valid2 = validating(valid_dataloader,p_valid,model,fold)
                valpreds222 = p_valid2[cols].values

                np.save(f"{cfg.savevalid}/valid{fold}_epoch{epoch}_num{valcount}",valpreds222)

                if cfg.fulltrain:
                  if valcount == cfg.stopvalidcount:
                    state = {
                                'state_dict': model.state_dict(),
                              #  'optimizer_dict': optimizer.state_dict(),
                                "bestscore":bestscore
                            }


                    torch.save(state, f"modelno{cfg.modelno}.pth")
                    p_valid2.to_csv(f"{cfg.savepath}/valid{cfg.fold}_seed{cfg.seed}.csv",index=False)

                    del state
                    torch.cuda.empty_cache()
                    gc.collect()

                    print("fulltrain save done")

                    break



                valcount+=1




                model.train()

                #### Early stop ####

                if bestscore > score:

                    print(f"Best score is {bestscore} → {score}. Saving model")
                    bestscore = score
                    bestlossmean = lossmean

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




                else:
                    print(f"no improvement. best is {bestscore}. score is {score}")

                    bestlossmean = lossmean

        else:
            bestlossmean = 0 # validationのlossを出力する
            valid_forscore = pd.DataFrame()
            p_valid2 = pd.DataFrame()




    allpreds = np.concatenate(allpreds)
    alltargets = np.concatenate(alltargets)

    score = get_score(alltargets,allpreds )

    print(score)

    losses = np.mean(losses)

    del logits,loss,ids,mask,targets,metric
    torch.cuda.empty_cache()
    gc.collect()



    return allpreds,losses,score,model,bestscore,bestlossmean,p_valid2