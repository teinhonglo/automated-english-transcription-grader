import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

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
            elif score_loss == "mcrmse":
                criterion = compute_mcrmse
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
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = (all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * compute_rmse(score_predictions, score_targets)
    
    return score_rmse
