import os
import numpy as np
import torch
import scipy.io
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numba
from deepgaze_pytorch.deepgaze2e import DeepGazeIIE

# Constantes
METRICS_SET_PATH = r"C:\Users\Adrian\Downloads\Data DeepGaze\CAT2000\metricsSet"
MODEL_PATH = r"C:\Users\Adrian\Desktop\deepgaze_pytorch\deepgaze_pytorch\checkpoints\final.pth"
RESULTS_PATH = r"C:\Users\Adrian\Desktop\deepgaze_pytorch\deepgaze_pytorch\checkpoints"
CENTER_BIAS_CACHE = os.path.join(RESULTS_PATH, "center_bias_cache.npy")

CATEGORY_MAPPING = {
    'Action': 0, 'Affective': 1, 'Art': 2, 'BlackWhite': 3,
    'Cartoon': 4, 'Fractal': 5, 'Indoor': 6, 'Inverted': 7,
    'Jumbled': 8, 'LineDrawing': 9, 'LowResolution': 10,
    'Noisy': 11, 'Object': 12, 'OutdoorManMade': 13,
    'OutdoorNatural': 14, 'Pattern': 15, 'Random': 16,
    'Satelite': 17, 'Sketch': 18, 'Social': 19
}

class CenterBiasModel:
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self._cached_log_density = None
        self._cached_shape = None
        
    def log_density(self, shape):
        """
        Calcula el center bias para una imagen de shape dado
        """
        if self._cached_log_density is not None and self._cached_shape == shape:
            return self._cached_log_density
        
        height, width = shape
        y = np.linspace(-1, 1, height)[:, np.newaxis]
        x = np.linspace(-1, 1, width)[np.newaxis, :]
        gaussian = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        gaussian = gaussian / gaussian.sum()
        log_density = np.log(gaussian + 1e-20)
        log_density = log_density - scipy.special.logsumexp(log_density)
        
        self._cached_log_density = log_density
        self._cached_shape = shape
        
        return log_density


class FullShuffledNonfixationProvider:
    """
    Implementación fiel al código original para shuffled fixations
    """
    def __init__(self, stimuli_shapes):
        # Guardamos las dimensiones de todas las imágenes
        self.heights = np.asarray([shape[0] for shape in stimuli_shapes]).astype(float)
        self.widths = np.asarray([shape[1] for shape in stimuli_shapes]).astype(float)
    
    def __call__(self, fixations, current_n):
        """
        Obtiene fijaciones shuffled para una imagen específica
        Args:
            fixations: objeto con todas las fijaciones
            current_n: índice de la imagen actual
        """
        # Obtener fijaciones de todas las otras imágenes
        other_img_inds = ~(fixations.n == current_n)
        xs = fixations.x[other_img_inds].copy().astype(float)
        ys = fixations.y[other_img_inds].copy().astype(float)
        
        # Escalar coordenadas según las dimensiones relativas
        other_ns = fixations.n[other_img_inds]
        xs *= self.widths[current_n]/self.widths[other_ns]
        ys *= self.heights[current_n]/self.heights[other_ns]
        
        return xs.astype(int), ys.astype(int)

# Funciones de métricas
def convert_saliency_map_to_density(saliency_map, minimum_value=1e-20):
    if saliency_map.min() < 0:
        saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map + minimum_value
    saliency_map_sum = saliency_map.sum()
    if saliency_map_sum:
        saliency_map = saliency_map / saliency_map_sum
    else:
        saliency_map[:] = 1.0
        saliency_map /= saliency_map.sum()
    return saliency_map

def normalize_for_cc(saliency_map):
    """
    Normalización específica para CC siguiendo la implementación original
    """
    saliency_map = np.asarray(saliency_map, dtype=float)
    saliency_map -= saliency_map.mean()
    std = saliency_map.std()
    if std:
        saliency_map /= std
    return saliency_map, std == 0

def debug_cc_calculation(saliency_map, fixation_map, category=None):
    """
    Debug detallado del cálculo de CC
    """
    print(f"\n=== Debug CC {'para ' + category if category else ''} ===")
    
    # 1. Estadísticas de entrada
    print("\nEstadísticas iniciales:")
    print("Saliency Map:")
    print(f"Shape: {saliency_map.shape}")
    print(f"Min: {saliency_map.min():.6f}")
    print(f"Max: {saliency_map.max():.6f}")
    print(f"Mean: {saliency_map.mean():.6f}")
    print(f"Std: {saliency_map.std():.6f}")
    print("\nFixation Map:")
    print(f"Shape: {fixation_map.shape}")
    print(f"Min: {fixation_map.min():.6f}")
    print(f"Max: {fixation_map.max():.6f}")
    print(f"Mean: {fixation_map.mean():.6f}")
    print(f"Std: {fixation_map.std():.6f}")
    print(f"Número de fijaciones: {(fixation_map > 0).sum()}")
    
    # 2. Debug normalización
    smap, constant1 = normalize_for_cc(saliency_map.copy())
    fmap, constant2 = normalize_for_cc(fixation_map.copy())
    
    print("\nDespués de normalización:")
    print("Saliency Map Normalizado:")
    print(f"Constante: {constant1}")
    print(f"Min: {smap.min():.6f}")
    print(f"Max: {smap.max():.6f}")
    print(f"Mean: {smap.mean():.6f}")
    print(f"Std: {smap.std():.6f}")
    print("\nFixation Map Normalizado:")
    print(f"Constante: {constant2}")
    print(f"Min: {fmap.min():.6f}")
    print(f"Max: {fmap.max():.6f}")
    print(f"Mean: {fmap.mean():.6f}")
    print(f"Std: {fmap.std():.6f}")
    
    # 3. Verificar valores inválidos
    print("\nVerificación de valores inválidos:")
    print(f"Saliency NaNs: {np.isnan(smap).sum()}")
    print(f"Saliency Infs: {np.isinf(smap).sum()}")
    print(f"Fixation NaNs: {np.isnan(fmap).sum()}")
    print(f"Fixation Infs: {np.isinf(fmap).sum()}")
    
    # 4. Debug correlación
    smap_flat = smap.flatten()
    fmap_flat = fmap.flatten()
    correlation = np.corrcoef(smap_flat, fmap_flat)[0, 1]
    print(f"\nCorrelación final: {correlation:.6f}")
    
    return {
        'correlation': correlation,
        'smap_stats': {
            'min': smap.min(),
            'max': smap.max(),
            'mean': smap.mean(),
            'std': smap.std()
        },
        'fmap_stats': {
            'min': fmap.min(),
            'max': fmap.max(),
            'mean': fmap.mean(),
            'std': fmap.std()
        }
    }

# def prepare_fixation_map(fixation_map):
#     """
#     Versión corregida de preparación del mapa de fijaciones
#     """
#     # Asegurar que es float y copiar
#     fmap = fixation_map.astype(float).copy()
    
#     # Contar número total de fijaciones
#     n_fixations = fmap.sum()
    
#     if n_fixations > 0:
#         # Normalizar por número de fijaciones
#         fmap /= n_fixations
#     else:
#         # Distribución uniforme si no hay fijaciones
#         fmap.fill(1.0 / (fmap.shape[0] * fmap.shape[1]))
    
#     # Evitar valores cero añadiendo un epsilon pequeño
#     eps = np.finfo(float).eps
#     fmap += eps
#     fmap /= fmap.sum()  # Renormalizar después de añadir epsilon
    
#     return fmap

def prepare_fixation_map(fixation_map):
    """
    Nueva versión con normalización ajustada
    """
    fmap = fixation_map.astype(float).copy()
    
    # Aplicar suavizado gaussiano para crear una densidad
    from scipy.ndimage import gaussian_filter
    fmap = gaussian_filter(fmap, sigma=10)
    
    # Normalizar a suma 1
    if fmap.sum() > 0:
        fmap /= fmap.sum()
    else:
        fmap.fill(1.0 / (fmap.shape[0] * fmap.shape[1]))
    
    return fmap

def debug_cc_calculation(saliency_map, fixation_map, category=None):
    """
    Debug detallado del cálculo de CC
    """
    print(f"\n=== Debug CC {'para ' + category if category else ''} ===")
    
    # Debug del fixation map antes de preparación
    print("\nFixation Map Original:")
    print(f"Shape: {fixation_map.shape}")
    print(f"Suma total: {fixation_map.sum()}")
    print(f"Número de fijaciones: {(fixation_map > 0).sum()}")
    
    # Preparar fixation map
    prepared_fmap = prepare_fixation_map(fixation_map)
    
    print("\nFixation Map Preparado:")
    print(f"Min: {prepared_fmap.min():.8f}")
    print(f"Max: {prepared_fmap.max():.8f}")
    print(f"Media: {prepared_fmap.mean():.8f}")
    print(f"Suma: {prepared_fmap.sum():.8f}")  # Debería ser cercano a 1
    
    # Normalizar mapas
    smap, constant1 = normalize_for_cc(saliency_map.copy())
    fmap, constant2 = normalize_for_cc(prepared_fmap)
    
    print("\nDespués de normalización:")
    print("Saliency Map Normalizado:")
    print(f"Constante: {constant1}")
    print(f"Min: {smap.min():.6f}")
    print(f"Max: {smap.max():.6f}")
    print(f"Mean: {smap.mean():.6f}")
    print(f"Std: {smap.std():.6f}")
    
    print("\nFixation Map Normalizado:")
    print(f"Constante: {constant2}")
    print(f"Min: {fmap.min():.6f}")
    print(f"Max: {fmap.max():.6f}")
    print(f"Mean: {fmap.mean():.6f}")
    print(f"Std: {fmap.std():.6f}")
    
    # Calcular correlación
    correlation = np.corrcoef(smap.flatten(), fmap.flatten())[0, 1]
    print(f"\nCorrelación final: {correlation:.6f}")
    
    return correlation

def calculate_cc(saliency_map, fixation_map, debug=False, category=None):
    """
    Calcula CC usando la preparación corregida del fixation map
    """
    if debug:
        return debug_cc_calculation(saliency_map, fixation_map, category)
        
    # Preparar fixation map
    fmap = prepare_fixation_map(fixation_map)
    
    # Normalizar mapas
    smap, constant1 = normalize_for_cc(saliency_map.copy())
    fmap, constant2 = normalize_for_cc(fmap)
    
    if constant1 and not constant2:
        return 0.0
    
    return np.corrcoef(smap.flatten(), fmap.flatten())[0, 1]


@numba.jit(nopython=True)
def _auc_for_one_positive(positive, negatives):
    """Calcula AUC para un valor positivo contra muchos negativos"""
    count = 0
    for negative in negatives:
        if negative < positive:
            count += 1
        elif negative == positive:
            count += 0.5
    return count / len(negatives)

def calculate_auc(saliency_map, fixation_map):
    """Calcula AUC tradicional"""
    positives = saliency_map[fixation_map > 0].ravel()
    if len(positives) == 0:
        return np.nan
    negatives = saliency_map.ravel()
    return _auc_for_one_positive(positives.mean(), negatives)

def calculate_sauc(saliency_map, fixation_map, nonfixation_provider, n, fixations, debug=False, category=None):
    """
    Calcula sAUC con opción de debug
    """
    if debug:
        debug_info = debug_sauc_calculation(saliency_map, fixation_map, nonfixation_provider, n, fixations, category)
    
    # No modificar el mapa de saliencia para sAUC
    positives = saliency_map[fixation_map > 0].ravel()
    if len(positives) == 0:
        return np.nan
    
    nonfix_xs, nonfix_ys = nonfixation_provider(fixations, n)
    valid = (nonfix_xs >= 0) & (nonfix_xs < saliency_map.shape[1]) & \
            (nonfix_ys >= 0) & (nonfix_ys < saliency_map.shape[0])
    
    if not valid.sum():
        return np.nan
    
    nonfix_xs = nonfix_xs[valid]
    nonfix_ys = nonfix_ys[valid]
    negatives = saliency_map[nonfix_ys, nonfix_xs]
    
    return _auc_for_one_positive(positives.mean(), negatives)

# def calculate_sauc(saliency_map, fixation_map, center_bias_template):
#     center_bias_template = center_bias_template / center_bias_template.max()
#     positives = saliency_map[fixation_map > 0].ravel()
#     negatives = saliency_map[center_bias_template > 0].ravel()
#     return _auc_for_one_positive(positives.mean(), negatives)

def debug_sauc_calculation(saliency_map, fixation_map, nonfixation_provider, n, fixations, category=None):
    """
    Debug detallado del cálculo de sAUC
    """
    print(f"\n=== Debug sAUC {'para ' + category if category else ''} ===")
    
    # 1. Estadísticas del mapa de saliencia
    print("\nEstadísticas del mapa de saliencia:")
    print(f"Shape: {saliency_map.shape}")
    print(f"Min: {saliency_map.min():.6f}")
    print(f"Max: {saliency_map.max():.6f}")
    print(f"Mean: {saliency_map.mean():.6f}")
    print(f"Std: {saliency_map.std():.6f}")
    
    # 2. Analizar fijaciones positivas
    positives = saliency_map[fixation_map > 0].ravel()
    print(f"\nFijaciones positivas:")
    print(f"Número: {len(positives)}")
    if len(positives) > 0:
        print(f"Media: {positives.mean():.6f}")
        print(f"Min: {positives.min():.6f}")
        print(f"Max: {positives.max():.6f}")
    
    # 3. Analizar fijaciones negativas (shuffled)
    nonfix_xs, nonfix_ys = nonfixation_provider(fixations, n)
    valid = (nonfix_xs >= 0) & (nonfix_xs < saliency_map.shape[1]) & \
            (nonfix_ys >= 0) & (nonfix_ys < saliency_map.shape[0])
    
    print(f"\nFijaciones negativas (shuffled):")
    print(f"Total propuesto: {len(nonfix_xs)}")
    print(f"Total válido: {valid.sum()}")
    
    if valid.sum() > 0:
        nonfix_xs = nonfix_xs[valid]
        nonfix_ys = nonfix_ys[valid]
        negatives = saliency_map[nonfix_ys, nonfix_xs]
        print(f"Media: {negatives.mean():.6f}")
        print(f"Min: {negatives.min():.6f}")
        print(f"Max: {negatives.max():.6f}")
        
        # 4. Análisis de distribución espacial
        center_y, center_x = saliency_map.shape[0]//2, saliency_map.shape[1]//2
        pos_y, pos_x = np.where(fixation_map > 0)
        pos_distances = np.sqrt((pos_y - center_y)**2 + (pos_x - center_x)**2)
        neg_distances = np.sqrt((nonfix_ys - center_y)**2 + (nonfix_xs - center_x)**2)
        
        print("\nDistribución espacial:")
        print(f"Distancia media al centro (positivos): {pos_distances.mean():.2f}")
        print(f"Distancia media al centro (negativos): {neg_distances.mean():.2f}")
    
    score = _auc_for_one_positive(positives.mean(), negatives)
    print(f"\nsAUC calculado: {score:.6f}")
    
    return {
        'n_positives': len(positives),
        'n_negatives': valid.sum(),
        'mean_positive': positives.mean() if len(positives) > 0 else None,
        'mean_negative': negatives.mean() if valid.sum() > 0 else None,
        'sauc_score': score
    }

# def debug_sauc_calculation(saliency_map, fixation_map, nonfixation_provider, n, fixations, category=None):
#     """
#     Función de diagnóstico para el cálculo de sAUC
#     """
#     print(f"\n=== Debug sAUC {'para ' + category if category else ''} ===")
    
#     # 1. Analizar distribución del mapa de saliencia
#     print("\nEstadísticas del mapa de saliencia:")
#     print(f"Shape: {saliency_map.shape}")
#     print(f"Min: {saliency_map.min():.4f}")
#     print(f"Max: {saliency_map.max():.4f}")
#     print(f"Mean: {saliency_map.mean():.4f}")
    
#     # 2. Analizar fijaciones positivas
#     positives = saliency_map[fixation_map > 0].ravel()
#     print(f"\nFijaciones positivas:")
#     print(f"Número de fijaciones: {len(positives)}")
#     if len(positives) > 0:
#         print(f"Media de valores positivos: {positives.mean():.4f}")
    
#     # 3. Analizar fijaciones negativas (shuffled)
#     nonfix_xs, nonfix_ys = nonfixation_provider.get_shuffled_nonfixations(fixations, n)
#     valid = (nonfix_xs >= 0) & (nonfix_xs < saliency_map.shape[1]) & \
#             (nonfix_ys >= 0) & (nonfix_ys < saliency_map.shape[0])
#     print(f"\nFijaciones negativas (shuffled):")
#     print(f"Total de coordenadas: {len(nonfix_xs)}")
#     print(f"Coordenadas válidas: {valid.sum()}")
    
#     if valid.sum() > 0:
#         nonfix_xs = nonfix_xs[valid]
#         nonfix_ys = nonfix_ys[valid]
#         negatives = saliency_map[nonfix_ys, nonfix_xs]
#         print(f"Media de valores negativos: {negatives.mean():.4f}")
        
#         # 4. Analizar distribución espacial
#         center_y, center_x = saliency_map.shape[0]//2, saliency_map.shape[1]//2
#         pos_distances = np.sqrt((fixation_map > 0).nonzero()[0] - center_y)**2 + \
#                        ((fixation_map > 0).nonzero()[1] - center_x)**2
#         neg_distances = np.sqrt((nonfix_ys - center_y)**2 + (nonfix_xs - center_x)**2)
        
#         print("\nDistribución espacial:")
#         print(f"Distancia media al centro (positivos): {pos_distances.mean():.2f}")
#         print(f"Distancia media al centro (negativos): {neg_distances.mean():.2f}")
    
#     # 5. Verificar escalado
#     original_shape = saliency_map.shape
#     scaled_xs = nonfix_xs / nonfixation_provider.widths[n] * original_shape[1]
#     scaled_ys = nonfix_ys / nonfixation_provider.heights[n] * original_shape[0]
    
#     print("\nVerificación de escalado:")
#     print(f"Shape original: {original_shape}")
#     print(f"Rango X escalado: {scaled_xs.min():.2f} - {scaled_xs.max():.2f}")
#     print(f"Rango Y escalado: {scaled_ys.min():.2f} - {scaled_ys.max():.2f}")
    
#     return {
#         'n_positives': len(positives),
#         'n_negatives': valid.sum(),
#         'mean_positive': positives.mean() if len(positives) > 0 else None,
#         'mean_negative': negatives.mean() if valid.sum() > 0 else None,
#         'pos_dist_to_center': pos_distances.mean(),
#         'neg_dist_to_center': neg_distances.mean(),
#     }

# def calculate_sauc(saliency_map, fixation_map, nonfixation_provider, n, fixations, debug=False, category=None):
#     """
#     Versión actualizada de calculate_sauc con diagnósticos
#     """
#     # Normalizar mapa de saliencia primero
#     smap = saliency_map.copy()
#     smap = (smap - smap.min()) / (smap.max() - smap.min())  # Normalizar a [0,1]
    
#     if debug:
#         debug_info = debug_sauc_calculation(smap, fixation_map, nonfixation_provider, n, fixations, category)
    
#     positives = smap[fixation_map > 0].ravel()
#     if len(positives) == 0:
#         return np.nan
        
#     nonfix_xs, nonfix_ys = nonfixation_provider.get_shuffled_nonfixations(fixations, n)
#     valid = (nonfix_xs >= 0) & (nonfix_xs < smap.shape[1]) & \
#             (nonfix_ys >= 0) & (nonfix_ys < smap.shape[0])
    
#     if not valid.sum():
#         return np.nan
        
#     nonfix_xs = nonfix_xs[valid]
#     nonfix_ys = nonfix_ys[valid]
#     negatives = smap[nonfix_ys, nonfix_xs]
    
#     return _auc_for_one_positive(positives.mean(), negatives)
        

def convert_saliency_map_to_density(saliency_map, minimum_value=1e-20):
    """
    Convierte mapa de saliencia a una distribución de densidad de probabilidad
    """
    if saliency_map.min() < 0:
        saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map + minimum_value
    
    saliency_map_sum = saliency_map.sum()
    if saliency_map_sum:
        saliency_map = saliency_map / saliency_map_sum
    else:
        saliency_map[:] = 1.0
        saliency_map /= saliency_map.sum()
    
    return saliency_map

def probabilistic_image_based_kl_divergence(logp1, logp2, log_regularization=0, quotient_regularization=0):
    """
    Calcula la divergencia KL entre dos distribuciones en espacio log
    """
    if log_regularization or quotient_regularization:
        return (np.exp(logp2) * np.log(log_regularization + np.exp(logp2) / 
               (np.exp(logp1) + quotient_regularization))).sum()
    else:
        return (np.exp(logp2) * (logp2 - logp1)).sum()

def calculate_kld(saliency_map, fixation_map, debug=False, category=None):
    """
    Implementación de KLD siguiendo MIT Saliency Benchmark
    """
    if debug:
        print(f"\n=== Debug KLD {'para ' + category if category else ''} ===")
        
        print("\nEstadísticas iniciales:")
        print("Saliency Map:")
        print(f"Shape: {saliency_map.shape}")
        print(f"Min: {saliency_map.min():.6f}")
        print(f"Max: {saliency_map.max():.6f}")
        print(f"Mean: {saliency_map.mean():.6f}")
        
        print("\nFixation Map:")
        print(f"Shape: {fixation_map.shape}")
        print(f"Min: {fixation_map.min():.6f}")
        print(f"Max: {fixation_map.max():.6f}")
        print(f"Mean: {fixation_map.mean():.6f}")
        print(f"Suma: {fixation_map.sum()}")
    
    # Convertir a densidades
    smap_density = convert_saliency_map_to_density(saliency_map.copy(), minimum_value=0)
    fmap_density = convert_saliency_map_to_density(fixation_map.copy(), minimum_value=0)
    
    if debug:
        print("\nDespués de conversión a densidad:")
        print("Saliency Density:")
        print(f"Min: {smap_density.min():.8f}")
        print(f"Max: {smap_density.max():.8f}")
        print(f"Mean: {smap_density.mean():.8f}")
        print(f"Sum: {smap_density.sum():.8f}")
        
        print("\nFixation Density:")
        print(f"Min: {fmap_density.min():.8f}")
        print(f"Max: {fmap_density.max():.8f}")
        print(f"Mean: {fmap_density.mean():.8f}")
        print(f"Sum: {fmap_density.sum():.8f}")
    
    # Convertir a log-densidad
    eps = 2.2204e-16  # valor usado en MIT benchmark
    log_smap = np.log(smap_density + eps)
    log_fmap = np.log(fmap_density + eps)
    
    if debug:
        print("\nDespués de log transform:")
        print("Log Saliency:")
        print(f"Min: {log_smap.min():.4f}")
        print(f"Max: {log_smap.max():.4f}")
        print(f"Mean: {log_smap.mean():.4f}")
        
        print("\nLog Fixation:")
        print(f"Min: {log_fmap.min():.4f}")
        print(f"Max: {log_fmap.max():.4f}")
        print(f"Mean: {log_fmap.mean():.4f}")
    
    # Calcular KLD
    kld = probabilistic_image_based_kl_divergence(
        log_smap, log_fmap,
        log_regularization=eps,
        quotient_regularization=eps
    )
    
    if debug:
        print(f"\nKLD calculado: {kld:.6f}")
    
    return kld

def calculate_sim(saliency_map, fixation_map, sigma=20, debug=False, category=None):
    """
    Calcula SIM con sigma ajustado y normalización previa
    """
    if debug:
        print(f"\n=== Debug SIM {'para ' + category if category else ''} ===")
        
        print("\nEstadísticas iniciales:")
        print("Saliency Map:")
        print(f"Shape: {saliency_map.shape}")
        print(f"Min: {saliency_map.min():.6f}")
        print(f"Max: {saliency_map.max():.6f}")
        print(f"Mean: {saliency_map.mean():.6f}")
        print(f"Sum: {saliency_map.sum():.6f}")
        
        print("\nFixation Map:")
        print(f"Shape: {fixation_map.shape}")
        print(f"Min: {fixation_map.min():.6f}")
        print(f"Max: {fixation_map.max():.6f}")
        print(f"Mean: {fixation_map.mean():.6f}")
        print(f"Sum: {fixation_map.sum():.6f}")
        print(f"Número de fijaciones: {(fixation_map > 0).sum()}")
    
    # 1. Aplicar blur gaussiano con sigma aumentado
    from scipy.ndimage import gaussian_filter
    fixation_map_blurred = gaussian_filter(fixation_map.astype(float), sigma=sigma)
    
    if debug:
        print("\nDespués de aplicar blur (sigma={:.1f}):".format(sigma))
        print("Fixation Map Blurred:")
        print(f"Min: {fixation_map_blurred.min():.8f}")
        print(f"Max: {fixation_map_blurred.max():.8f}")
        print(f"Mean: {fixation_map_blurred.mean():.8f}")
        print(f"Sum: {fixation_map_blurred.sum():.8f}")
    
    # 2. Normalizar ambos mapas al mismo rango [0,1]
    smap_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    fmap_norm = fixation_map_blurred / fixation_map_blurred.max()
    
    if debug:
        print("\nDespués de normalización al rango [0,1]:")
        print("Saliency Map Normalizado:")
        print(f"Min: {smap_norm.min():.8f}")
        print(f"Max: {smap_norm.max():.8f}")
        print(f"Mean: {smap_norm.mean():.8f}")
        print(f"Sum: {smap_norm.sum():.8f}")
        
        print("\nFixation Map Normalizado:")
        print(f"Min: {fmap_norm.min():.8f}")
        print(f"Max: {fmap_norm.max():.8f}")
        print(f"Mean: {fmap_norm.mean():.8f}")
        print(f"Sum: {fmap_norm.sum():.8f}")
    
    # 3. Convertir a densidades
    smap_density = convert_saliency_map_to_density(smap_norm, minimum_value=0)
    fmap_density = convert_saliency_map_to_density(fmap_norm, minimum_value=0)
    
    if debug:
        print("\nDespués de conversión a densidad:")
        print("Saliency Density:")
        print(f"Min: {smap_density.min():.8f}")
        print(f"Max: {smap_density.max():.8f}")
        print(f"Mean: {smap_density.mean():.8f}")
        print(f"Sum: {smap_density.sum():.8f}")
        
        print("\nFixation Density:")
        print(f"Min: {fmap_density.min():.8f}")
        print(f"Max: {fmap_density.max():.8f}")
        print(f"Mean: {fmap_density.mean():.8f}")
        print(f"Sum: {fmap_density.sum():.8f}")
        
        # Analizar el mínimo elemento a elemento
        min_density = np.minimum(smap_density, fmap_density)
        print("\nEstadísticas del mínimo:")
        print(f"Min: {min_density.min():.8f}")
        print(f"Max: {min_density.max():.8f}")
        print(f"Mean: {min_density.mean():.8f}")
        print(f"Sum: {min_density.sum():.8f}")
    
    # Calcular SIM
    similarity = np.minimum(smap_density, fmap_density).sum()
    
    if debug:
        print(f"\nSIM calculado: {similarity:.6f}")
    
    return similarity

# def calculate_ig(saliency_map, fixation_map, baseline_log_likelihood):
#     log_density = np.log(convert_saliency_map_to_density(saliency_map))
#     fixation_points = np.where(fixation_map > 0)
#     current_log_likelihood = log_density[fixation_points].mean()
#     return current_log_likelihood - baseline_log_likelihood
def prepare_probabilistic_map(saliency_map, center_bias=None, eps=1e-20):
    """
    Prepara un mapa de saliencia como distribución probabilística
    incluyendo center prior y regularización apropiada
    """
    # Normalizar a valores positivos
    if saliency_map.min() < 0:
        saliency_map = saliency_map - saliency_map.min()
    
    # Aplicar regularización
    saliency_map += eps
    
    # Incorporar center prior si está disponible
    if center_bias is not None:
        # Convertir center_bias a probabilidad
        center_bias = np.exp(center_bias)
        center_bias = center_bias / center_bias.sum()
        
        # Combinar con el mapa de saliencia (multiplicación punto a punto)
        saliency_map = saliency_map * center_bias
    
    # Normalizar a distribución de probabilidad
    saliency_map = saliency_map / saliency_map.sum()
    
    return saliency_map

def calculate_ig(saliency_map, fixation_map, baseline_log_density, center_bias=None, debug=False, category=None):
    """
    Calcula Information Gain siguiendo las especificaciones del paper
    """
    if debug:
        print(f"\n=== Debug IG {'para ' + category if category else ''} ===")
        print("\nEstadísticas iniciales:")
        print(f"Saliency map min/max: {saliency_map.min():.6f}, {saliency_map.max():.6f}")
        print(f"Número de fijaciones: {(fixation_map > 0).sum()}")
        print(f"Baseline log-likelihood: {baseline_log_density:.6f}")
    
    # Preparar mapa probabilístico
    smap_prob = prepare_probabilistic_map(saliency_map, center_bias)
    
    if debug:
        print("\nDespués de preparación probabilística:")
        print(f"Min: {smap_prob.min():.8f}")
        print(f"Max: {smap_prob.max():.8f}")
        print(f"Sum: {smap_prob.sum():.8f}")
    
    # Convertir a log-densidad
    log_density = np.log(smap_prob)
    
    # Obtener puntos de fijación
    fixation_points = np.where(fixation_map > 0)
    if len(fixation_points[0]) == 0:
        return np.nan
    
    # Calcular log-likelihood en puntos de fijación
    log_likelihoods = log_density[fixation_points]
    model_log_likelihood = log_likelihoods.mean()
    
    # Calcular IG
    ig = model_log_likelihood - baseline_log_density
    
    if debug:
        print("\nEstadísticas finales:")
        print(f"Model log-likelihood: {model_log_likelihood:.6f}")
        print(f"Baseline log-likelihood: {baseline_log_density:.6f}")
        print(f"IG calculado: {ig:.6f}")
        
        # Análisis adicional
        print("\nAnálisis de predicciones:")
        print(f"Predicciones mejores que baseline: {(log_likelihoods > baseline_log_density).mean()*100:.2f}%")
        
        if center_bias is not None:
            print("\nEstadísticas con center prior:")
            print(f"Center prior influencia: {np.corrcoef(smap_prob.flatten(), np.exp(center_bias).flatten())[0,1]:.4f}")
    
    return ig

def calculate_nss(saliency_map, fixation_map, debug=False, category=None):
    """
    Calcula NSS (Normalized Scanpath Saliency)
    """
    if debug:
        print(f"\n=== Debug NSS {'para ' + category if category else ''} ===")
        
        print("\nEstadísticas iniciales:")
        print(f"Número de fijaciones: {(fixation_map > 0).sum()}")
        print(f"Saliency min/max: {saliency_map.min():.6f}, {saliency_map.max():.6f}")
    
    # Normalizar el mapa de saliencia
    smap = (saliency_map - saliency_map.mean()) / saliency_map.std()
    
    # Obtener valores de saliencia en puntos de fijación
    fixation_points = fixation_map > 0
    if not fixation_points.any():
        return np.nan
    
    nss = smap[fixation_points].mean()
    
    if debug:
        print("\nEstadísticas después de normalización:")
        print(f"Saliency mean: {smap.mean():.6f}")
        print(f"Saliency std: {smap.std():.6f}")
        print(f"NSS calculado: {nss:.6f}")
    
    return nss

def load_or_create_center_bias(category_data, cache_path):
    """Carga o crea center bias por categoría"""
    if os.path.exists(cache_path):
        print("Cargando center bias cache...")
        return np.load(cache_path, allow_pickle=True).item()
    
    print("Creando nuevo center bias para cada categoría...")
    center_bias_dict = {}
    cbm = CenterBiasModel(sigma=0.5)
    
    for category, data in category_data.items():
        if not data:
            continue
        
        # Usar primera imagen para obtener dimensiones
        sample_image = Image.open(data[0]['image_path'])
        center_bias = cbm.log_density(np.array(sample_image))
        center_bias_dict[category] = center_bias
        
        print(f"\nCenter Bias stats para {category}:")
        print(f"Shape: {center_bias.shape}")
        print(f"Min: {center_bias.min():.4f}")
        print(f"Max: {center_bias.max():.4f}")
    
    np.save(cache_path, center_bias_dict)
    return center_bias_dict

def create_center_bias_model(stimuli, fixations):
    """
    Calcula el baseline log-likelihood usando center bias
    """
    print("Calculando baseline log-likelihood...")
    cbm = CenterBiasModel(sigma=0.5)
    total_ll = 0.0
    count = 0
    
    for n in tqdm(range(len(stimuli))):
        stimulus = stimuli[n]
        fix_inds = fixations.n == n
        
        if not np.any(fix_inds):
            continue
            
        x = fixations.x[fix_inds]
        y = fixations.y[fix_inds]
        
        log_density = cbm.log_density(stimulus.shape[:2])
        current_ll = np.sum(log_density[y.astype(int), x.astype(int)])
        
        if len(x) > 0:
            total_ll += current_ll
            count += len(x)
    
    baseline_ll = total_ll / count if count > 0 else 0
    print(f"Baseline Log-Likelihood calculado: {baseline_ll:.4f}")
    
    return baseline_ll

def load_validation_data():
    """Carga datos de validación organizados por categoría"""
    data_by_category = defaultdict(list)
    
    stimuli_dir = os.path.join(METRICS_SET_PATH, 'Stimuli')
    fixations_dir = os.path.join(METRICS_SET_PATH, 'FixationLocs')
    
    for category in CATEGORY_MAPPING.keys():
        category_path = os.path.join(stimuli_dir, category)
        fixation_category_path = os.path.join(fixations_dir, category)
        
        if not os.path.isdir(category_path):
            print(f"Advertencia: No se encontró la carpeta {category}")
            continue
        
        print(f"\nCargando categoría: {category}")
        for img_name in os.listdir(category_path):
            if not img_name.endswith('.jpg'):
                continue
            
            base_name = os.path.splitext(img_name)[0]
            mat_name = f"{base_name}.mat"
            
            img_path = os.path.join(category_path, img_name)
            mat_path = os.path.join(fixation_category_path, mat_name)
            
            if os.path.exists(mat_path):
                data_by_category[category].append({
                    'image_path': img_path,
                    'fixation_path': mat_path,
                    'base_name': base_name
                })
    
    return data_by_category

def load_our_model(model_path, device, num_categories=20):
    print("Cargando modelo...")
    model = DeepGazeIIE(num_categories=num_categories, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_by_category, device='cuda', debug=True):
    """
    Evalúa el modelo con las métricas actualizadas
    """
    model.eval()
    results = defaultdict(lambda: defaultdict(list))
    
    # Crear CenterBiasModel
    cbm = CenterBiasModel(sigma=0.5)
    
    # Recolectar información de fijaciones y shapes
    print("\nRecolectando información de fijaciones...")
    all_fixations = defaultdict(list)
    stimulus_shapes = []
    
    # Primera pasada: recolectar todas las fijaciones y shapes
    global_idx = 0
    for category, data_list in data_by_category.items():
        for data in tqdm(data_list, desc=f"Procesando {category}"):
            fixation_data = scipy.io.loadmat(data['fixation_path'])
            fixation_map = fixation_data['fixLocs']
            ys, xs = np.where(fixation_map > 0)
            
            all_fixations['x'].extend(xs)
            all_fixations['y'].extend(ys)
            all_fixations['n'].extend([global_idx] * len(xs))
            
            img = Image.open(data['image_path'])
            stimulus_shapes.append(img.size[::-1])  # (height, width)
            
            global_idx += 1
    
    # Crear objeto de fijaciones y nonfixation provider
    fixations = type('Fixations', (), {
        'x': np.array(all_fixations['x']),
        'y': np.array(all_fixations['y']),
        'n': np.array(all_fixations['n'])
    })

        # Calcular baseline log-likelihood
    print("\nCalculando baseline log-likelihood...")
    cbm = CenterBiasModel(sigma=0.5)
    total_ll = 0.0
    count = 0
    
    for n, shape in enumerate(tqdm(stimulus_shapes)):
        fix_inds = fixations.n == n
        if not np.any(fix_inds):
            continue
            
        x = fixations.x[fix_inds]
        y = fixations.y[fix_inds]
        
        log_density = cbm.log_density(shape)
        current_ll = np.sum(log_density[y.astype(int), x.astype(int)])
        
        if len(x) > 0:
            total_ll += current_ll
            count += len(x)

    baseline_log_density = total_ll / count if count > 0 else 0
    print(f"Baseline Log-Likelihood calculado: {baseline_log_density:.4f}")
    
    nonfixation_provider = FullShuffledNonfixationProvider(stimulus_shapes)
    
    # Segunda pasada: evaluación
    print("\nEvaluando modelo...")
    global_idx = 0
    with torch.no_grad():
        for category, data_list in data_by_category.items():
            print(f"\nEvaluando categoría: {category}")
            category_idx = torch.tensor([CATEGORY_MAPPING[category]], device=device)
            
            # Debug solo primera imagen de cada categoría
            do_debug = debug and len(results[category]['CC']) == 0
            
            for idx, data in enumerate(tqdm(data_list)):
                # Cargar y preprocesar imagen
                image = Image.open(data['image_path']).convert('RGB')
                image_tensor = torch.FloatTensor(np.array(image)) / 255.0
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Calcular center bias
                center_bias = cbm.log_density(image.size[::-1])
                center_bias_tensor = torch.FloatTensor(center_bias).unsqueeze(0).to(device)
                
                # Cargar fijaciones
                fixation_data = scipy.io.loadmat(data['fixation_path'])
                fixation_map = fixation_data['fixLocs']
                
                # Generar predicción
                saliency_map = model(image_tensor, center_bias_tensor, category_idx)
                saliency_map = saliency_map.cpu().numpy().squeeze()
                
                # No modificar saliency_map original para sAUC
                saliency_map_orig = saliency_map.copy()
                
                metrics = {
                    'AUC': calculate_auc(saliency_map_orig, fixation_map),
                    'sAUC': calculate_sauc(saliency_map_orig, fixation_map, 
                                    nonfixation_provider, global_idx, fixations,
                                    debug=do_debug, category=category),
                    'CC': calculate_cc(saliency_map, fixation_map, 
                                debug=do_debug, category=category),
                    'KLD': calculate_kld(saliency_map, fixation_map,
                                    debug=do_debug, category=category),
                    'SIM': calculate_sim(saliency_map, fixation_map, sigma=20,
                         debug=do_debug, category=category),
                    'NSS': calculate_nss(saliency_map, fixation_map,
                         debug=do_debug, category=category),
                    'IG': calculate_ig(saliency_map, fixation_map,
                                       baseline_log_density=baseline_log_density,
                                         center_bias=center_bias, 
                                       debug=do_debug, category=category)
                }
                
                if do_debug:
                    print(f"\nResultados primera imagen de {category}:")
                    for metric_name, value in metrics.items():
                        print(f"{metric_name}: {value:.6f}")
                
                for metric_name, value in metrics.items():
                    results[category][metric_name].append(value)
                
                global_idx += 1
    
    # Calcular estadísticas
    summary = defaultdict(dict)
    for category in results:
        for metric in results[category]:
            values = np.array(results[category][metric])
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                summary[category][f'{metric}_mean'] = valid_values.mean()
                summary[category][f'{metric}_std'] = valid_values.std()
            else:
                summary[category][f'{metric}_mean'] = np.nan
                summary[category][f'{metric}_std'] = np.nan
    
    # Crear DataFrame final
    df_summary = pd.DataFrame.from_dict(summary, orient='index')
    global_means = df_summary.mean()
    global_summary = pd.DataFrame(global_means).T
    global_summary.index = ['GLOBAL']
    
    return pd.concat([df_summary, global_summary])

def main():
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsando device: {device}")
    
    # Cargar modelo
    model = load_our_model(MODEL_PATH, device)
    model.eval()
    
    # Cargar datos
    data_by_category = load_validation_data()
    
    # Evaluar modelo
    results = evaluate_model(model, data_by_category, device)
    
    # Guardar resultados
    results_path = os.path.join(RESULTS_PATH, "saliency_metrics_results_final.csv")
    results.to_csv(results_path)
    print(f"\nResultados guardados en: {results_path}")
    print("\nResumen de resultados:")
    print(results)

if __name__ == "__main__":
    main()