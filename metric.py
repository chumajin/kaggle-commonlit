
from util import *

def MCRMSE(y_trues, y_preds):
    scores = []

    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
        print(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    try:
      y_true = y_true.detach().cpu().numpy()
      y_pred = y_pred.detach().cpu().numpy()
    except:
      pass
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score

