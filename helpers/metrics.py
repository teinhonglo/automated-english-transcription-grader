import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import exp

class LNLoss(nn.Module):
    def __init__(self, power=1):
        super().__init__()
        self.loss = nn.L1Loss(reduction='mean')
        self.pow = power

    def forward(self, y_pred, y_true):
        loss = torch.pow(self.loss(y_pred, y_true), self.pow)
        return loss

def get_losses(training_objectives, training_objective_predictions, labels, device, score_loss):
    """
    Returns a dict containing training objectives and their respective losses over their predictions and the combined
    weighted loss of the training objectives.
    """
    losses = {'overall': torch.tensor([0.], device=device)}
    for objective, prediction in training_objective_predictions.items():
        _, alpha = training_objectives[objective]
        # Scoring objective uses MSE loss and all other objectives use CrossEntropy loss
        if objective == 'score':
            if score_loss == "mse":
                criterion = nn.MSELoss().to(device)
            elif score_loss == "rmse":
                criterion = compute_rmse
            elif score_loss == 'smooth_l1':
                criterion = nn.SmoothL1Loss().to(device)
            elif score_loss == 'ln':
                criterion = LNLoss(power=1.5).to(device)
            elif score_loss == "mcrmse":
                criterion = compute_mcrmse
            elif score_loss == "mcmse":
                criterion = compute_mcmse
            elif score_loss == "wmse":
                criterion = compute_wmse
            elif score_loss == "pearson":
                criterion = pearson_loss
            elif score_loss == "centroid":
                criterion = centroid_loss
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)

        batch_labels = labels[objective].reshape(-1)

        losses[objective] = criterion(prediction, batch_labels)
        losses['overall'] += (losses[objective] * alpha)

    return losses

def compute_metrics(total_losses, all_score_predictions, all_score_targets, device):
    """ Computes Pearson correlation and accuracy within 0.5 and 1 of target score and adds each to total_losses dict. """
    #total_losses['rmse'] = compute_rmse(all_score_predictions, all_score_targets).cpu()
    #total_losses['mcrmse'] = compute_mcrmse(all_score_predictions, all_score_targets).cpu()
    print("predictions", all_score_predictions)
    print("targets", all_score_targets)
    total_losses['pearson'] = stats.pearsonr(all_score_predictions.cpu(), all_score_targets.cpu())[0]
    total_losses['within_0.5'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 0.5,
                                                              device)
    total_losses['within_1'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 1,
                                                                device)

def _accuracy_within_margin(score_predictions, score_target, margin, device):
    """ Returns the percentage of predicted scores that are within the provided margin from the target score. """
    return torch.sum(
        torch.where(
            torch.abs(score_predictions - score_target) <= margin,
            torch.ones(len(score_predictions), device=device),
            torch.zeros(len(score_predictions), device=device))).item() / len(score_predictions) * 100

def compute_rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

def compute_mcrmse(all_score_predictions, all_score_targets):
    print(all_score_predictions.shape, all_score_targets.shape)
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = (all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * compute_rmse(score_predictions, score_targets)
    
    return score_rmse

def compute_mse(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def compute_mcmse(all_score_predictions, all_score_targets):
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_mse = 0.
    
    for c in unique_classes:
        indices = (all_score_targets == c)
        if len(indices) == 1:
            all_score_predictions = all_score_predictions.unsqueeze(0)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_mse += 1 / num_classes * compute_mse(score_predictions, score_targets)
    
    return score_mse

def compute_wmse(all_score_predictions, all_score_targets):
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_mse = 0.

    #class_weight = [0, 0.18, 0.18, 0.18, 0.16, 0.10, 0.10, 0.10]
    class_weight = [ 1 / (all_score_targets == c).sum().item() for c in unique_classes ]
    class_weight = softmax(class_weight)

    for i, c in enumerate(unique_classes):
        indices = (all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        #score_mse += class_weight[int(c.item())-1] / num_classes * compute_mse(score_predictions, score_targets)
        score_mse += class_weight[i] * compute_mse(score_predictions, score_targets)
    
    return score_mse

def softmax(x):
    return exp(x) / exp(x).sum()

def pearson(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_den = torch.clamp(r_den, min=1e-6)
    r_val = r_num / r_den
    r_val = torch.clamp(r_val, min=-1., max=1.)
    return r_val

def pearson_loss(pred, target):
    #return compute_mse(pred, target) + 0.5 * (1. - pearson(pred, target))
    return compute_mse(pred, target) - 1.0 * pearson(pred, target)

def centroid_loss(pred, target):
    return compute_mse(pred, target) + torch.abs(torch.mean(pred) - torch.mean(target))
