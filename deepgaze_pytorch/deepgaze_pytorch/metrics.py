#metrics.py

import numpy as np
from pysaliency.roc import general_roc
from pysaliency.numba_utils import auc_for_one_positive
import torch


import numpy as np
import torch
import torch.nn.functional as F

# def auc_for_one_positive(positive, negatives):
#     """Calcula el AUC para un solo punto positivo."""
#     return (negatives < positive).mean() + 0.5 * (negatives == positive).mean()

# def general_roc(positives, negatives):
#     """Calcula la curva ROC general."""
#     thresholds = np.concatenate((positives, negatives))
#     thresholds = np.sort(thresholds)
    
#     tp = np.zeros_like(thresholds)
#     fp = np.zeros_like(thresholds)
    
#     for i, thresh in enumerate(thresholds):
#         tp[i] = (positives >= thresh).mean()
#         fp[i] = (negatives >= thresh).mean()
    
#     # Agregar puntos extremos
#     tp = np.concatenate(([1.0], tp, [0.0]))
#     fp = np.concatenate(([1.0], fp, [0.0]))
    
#     return np.trapz(tp, fp), tp, fp

# def _general_auc(positives, negatives):
#     """Wrapper para calcular AUC."""
#     if len(positives) == 1:
#         return auc_for_one_positive(positives[0], negatives)
#     else:
#         return general_roc(positives, negatives)[0]

def _general_auc(positives, negatives):
    if len(positives) == 1:
        return auc_for_one_positive(positives[0], negatives)
    else:
        return general_roc(positives, negatives)[0]

# def log_likelihood(log_density, fixation_mask, weights=None):
#     """
#     Calcula el log-likelihood (Information Gain).
    
#     Args:
#         log_density: Tensor con la predicción del modelo en escala logarítmica
#         fixation_mask: Máscara de fijaciones (sparse o dense)
#         weights: Pesos para cada muestra en el batch
#     """
#     # if weights is None:
#     #     weights = torch.ones(log_density.shape[0], device=log_density.device)

#     # Normalizar pesos
#     weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

#     # Convertir máscara sparse a dense si es necesario
#     if isinstance(fixation_mask, torch.sparse.IntTensor):
#         dense_mask = fixation_mask.to_dense()
#     else:
#         dense_mask = fixation_mask

#     # Calcular número de fijaciones por imagen
#     fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    
#     # Calcular log-likelihood promedio
#     ll = torch.mean(
#         weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
#     )
    
#     # Normalizar por el tamaño de la imagen y convertir a bits
#     return (ll + np.log(log_density.shape[-1] * log_density.shape[-2])) / np.log(2)

# FUNCIONA
# def log_likelihood(log_density, fixation_mask, weights=None):
#     """
#     Calcula el log-likelihood normalizado.
#     """
#     weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    
#     if isinstance(fixation_mask, torch.sparse.IntTensor):
#         dense_mask = fixation_mask.to_dense()
#     else:
#         dense_mask = fixation_mask
    
#     fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    
#     # Calcular log-likelihood
#     ll = torch.mean(
#         weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
#     )
    
#     # Normalizar por tamaño de imagen
#     # ll = ll + np.log(log_density.shape[-1] * log_density.shape[-2])
    
#     # Convertir a bits (dividir por log(2))
#     ll = ll / np.log(2)
    
#     return ll

# FUNCIONA GAUSS
# def log_likelihood(log_density, fixation_mask, weights=None):
#     """
#     Implementación exacta del paper original
#     """
#     # Debug entrada
#     print("\nCalculando Log-Likelihood:")
#     print(f"Shape log_density: {log_density.shape}")
#     print(f"Shape fixation_mask: {fixation_mask.shape}")
#     print(f"Range log_density: [{log_density.min().item():.4f}, {log_density.max().item():.4f}]")

#     #Debug 
#     exp_sum = torch.exp(log_density).sum(dim=(-1,-2))
#     if not torch.allclose(exp_sum, torch.ones_like(exp_sum), rtol=1e-3):
#         print(f"Warning: log_density no está normalizado correctamente (sum exp = {exp_sum.mean().item():.4f})")
    
#     weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    
#     if isinstance(fixation_mask, torch.sparse.IntTensor):
#         dense_mask = fixation_mask.to_dense()
#     else:
#         dense_mask = fixation_mask
    
#     fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    
#     # Debug pre-cálculo
#     print(f"Número de fijaciones: {fixation_count.sum().item()}")
    
#     # Cálculo directo del log-likelihood
#     ll = torch.mean(
#         weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
#     )
    
#     ll_bits = ll / np.log(2)  # convertir a bits
    
#     # Debug resultado
#     print(f"LL (nats): {ll.item():.4f}")
#     print(f"LL (bits): {ll_bits.item():.4f}")
    
#     return ll_bits

# FUNCIONA GAUSS
# def log_likelihood(log_density, fixation_mask, weights=None):
#     """
#     Calcula el log-likelihood en nats.
#     """
#     weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    
#     if isinstance(fixation_mask, torch.sparse.IntTensor):
#         dense_mask = fixation_mask.to_dense()
#     else:
#         dense_mask = fixation_mask
    
#     fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    
#     # Calcular log-likelihood en nats
#     ll = torch.mean(
#         weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
#     )
    
#     # Debug información
#     # print("\nCalculando Log-Likelihood:")
#     # print(f"Shape log_density: {log_density.shape}")
#     # print(f"Shape fixation_mask: {fixation_mask.shape}")
#     # print(f"Range log_density: [{log_density.min().item():.4f}, {log_density.max().item():.4f}]")
#     # print(f"Número de fijaciones: {fixation_count.sum().item()}")
#     # print(f"LL (nats): {ll:.4f}")
    
#     return ll  # Retornamos directamente en nats

def log_likelihood(log_density, fixation_mask, weights=None, reduction='mean'):
    """
    Calcula el log likelihood de las fijaciones
    
    Args:
        log_density: Log de la densidad predicha
        fixation_mask: Máscara de fijaciones
        weights: Pesos opcionales para cada muestra
        reduction: Tipo de reducción ('mean', 'sum', 'none')
    """
    ll = (log_density * fixation_mask).sum(dim=(1, 2))
    
    if weights is not None:
        ll = ll * weights
    
    if reduction == 'none':
        return ll
    elif reduction == 'mean':
        return ll.mean()
    elif reduction == 'sum':
        return ll.sum()
    else:
        raise ValueError(f"Reducción no soportada: {reduction}")

def information_gain(log_density, fixation_mask, baseline_ll, weights=None):
    """
    Calcula el Information Gain normalizado.
    """
    current_ll = log_likelihood(log_density, fixation_mask, weights)
    baseline_bits = baseline_ll / np.log(2)  # Convertir baseline a bits también
    
    return current_ll - baseline_bits

def nss(log_density, fixation_mask, weights=None, reduction='mean'):
    """
    Calcula el Normalized Scanpath Saliency
    """
    density = torch.exp(log_density)
    
    # Normalizar la densidad por imagen
    mean = density.mean(dim=(1, 2), keepdim=True)
    std = density.std(dim=(1, 2), keepdim=True)
    normalized_density = (density - mean) / (std + 1e-8)
    
    # Calcular NSS por muestra
    nss_per_sample = (normalized_density * fixation_mask).sum(dim=(1, 2)) / (fixation_mask.sum(dim=(1, 2)) + 1e-8)
    
    if weights is not None:
        nss_per_sample = nss_per_sample * weights
    
    if reduction == 'none':
        return nss_per_sample
    elif reduction == 'mean':
        return nss_per_sample.mean()
    elif reduction == 'sum':
        return nss_per_sample.sum()
    else:
        raise ValueError(f"Reducción no soportada: {reduction}")

def auc(log_density, fixation_mask, weights=None, reduction='mean'):
    """
    Calcula el Area Under Curve
    """
    density = torch.exp(log_density)
    
    # Calcular AUC por muestra
    auc_per_sample = torch.zeros(len(density), device=density.device)
    
    for i in range(len(density)):
        # Calcular AUC para cada imagen
        d = density[i].flatten()
        f = fixation_mask[i].flatten()
        
        # Ordenar valores de densidad
        sort_idx = torch.argsort(d, descending=True)
        d = d[sort_idx]
        f = f[sort_idx]
        
        # Calcular TPR y FPR
        tp = torch.cumsum(f, dim=0)
        fp = torch.cumsum(1 - f, dim=0)
        
        # Normalizar
        tp = tp / (tp[-1] + 1e-8)
        fp = fp / (fp[-1] + 1e-8)
        
        # Calcular AUC usando regla del trapecio
        auc_per_sample[i] = torch.trapz(tp, fp)
    
    if weights is not None:
        auc_per_sample = auc_per_sample * weights
    
    if reduction == 'none':
        return auc_per_sample
    elif reduction == 'mean':
        return auc_per_sample.mean()
    elif reduction == 'sum':
        return auc_per_sample.sum()
    else:
        raise ValueError(f"Reducción no soportada: {reduction}")

# # FUNCIONA GAUSS
# def nss(log_density, fixation_mask, weights=None):
#     """
#     Calcula el Normalized Scanpath Saliency.
    
#     Args:
#         log_density: Tensor con la predicción del modelo en escala logarítmica
#         fixation_mask: Máscara de fijaciones (sparse o dense)
#         weights: Pesos para cada muestra en el batch
#     """
#     # if weights is None:
#     #     weights = torch.ones(log_density.shape[0], device=log_density.device)
    
#     weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

#     if isinstance(fixation_mask, torch.sparse.IntTensor):
#         dense_mask = fixation_mask.to_dense()
#     else:
#         dense_mask = fixation_mask

#     fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

#     # Convertir log_density a density y normalizar
#     density = torch.exp(log_density)
#     mean, std = torch.std_mean(density, dim=(-1, -2), keepdim=True)
#     saliency_map = (density - mean) / std

#     # Calcular NSS
#     nss_value = torch.mean(
#         weights * torch.sum(saliency_map * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
#     )
#     return nss_value

# # FUNCIONA GAUSS

# def auc(log_density, fixation_mask, weights=None):
#     weights = len(weights) * weights / weights.sum()

#     # TODO: This doesn't account for multiple fixations in the same location!
#     def image_auc(log_density, fixation_mask):
#         if isinstance(fixation_mask, torch.sparse.IntTensor):
#             dense_mask = fixation_mask.to_dense()
#         else:
#             dense_mask = fixation_mask

#         positives = torch.masked_select(log_density, dense_mask.type(torch.bool)).detach().cpu().numpy().astype(np.float64)
#         negatives = log_density.flatten().detach().cpu().numpy().astype(np.float64)

#         auc = _general_auc(positives, negatives)

#         return torch.tensor(auc)

#     return torch.mean(weights.cpu() * torch.tensor([
#         image_auc(log_density[i], fixation_mask[i]) for i in range(log_density.shape[0])
#     ]))

# def auc(log_density, fixation_mask, weights=None):
#     """
#     Calcula el Area Under the Curve (AUC).
    
#     Args:
#         log_density: Tensor con la predicción del modelo en escala logarítmica
#         fixation_mask: Máscara de fijaciones (sparse o dense)
#         weights: Pesos para cada muestra en el batch
#     """
#     # if weights is None:
#     #     weights = torch.ones(log_density.shape[0], device=log_density.device)
    
#     weights = len(weights) * weights / weights.sum()

#     def image_auc(log_density, fixation_mask):
#         if isinstance(fixation_mask, torch.sparse.IntTensor):
#             dense_mask = fixation_mask.to_dense()
#         else:
#             dense_mask = fixation_mask

#         positives = torch.masked_select(
#             log_density, 
#             dense_mask.bool()
#         ).detach().cpu().numpy().astype(np.float64)
        
#         negatives = log_density.flatten().detach().cpu().numpy().astype(np.float64)
#         return torch.tensor(_general_auc(positives, negatives))

#     # Calcular AUC para cada imagen en el batch
#     aucs = torch.stack([
#         image_auc(log_density[i], fixation_mask[i]) 
#         for i in range(log_density.shape[0])
#     ])
    
#     return torch.mean(weights.cpu() * aucs)