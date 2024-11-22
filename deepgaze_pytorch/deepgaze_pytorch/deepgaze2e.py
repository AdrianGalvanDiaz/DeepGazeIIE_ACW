
# from collections import OrderedDict
# import importlib
# import os


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch.utils import model_zoo

# from .modules import FeatureExtractor, Finalizer, DeepGazeIIIMixture, MixtureModel

# from .layers import (
#     Conv2dMultiInput,
#     LayerNorm,
#     LayerNormMultiInput,
#     Bias,
# )


# BACKBONES = [
#     {
#         'type': 'deepgaze_pytorch.features.shapenet.RGBShapeNetC',
#         'used_features': [
#             '1.module.layer3.0.conv2',
#             '1.module.layer3.3.conv2',
#             '1.module.layer3.5.conv1',
#             '1.module.layer3.5.conv2',
#             '1.module.layer4.1.conv2',
#             '1.module.layer4.2.conv2',
#         ],
#         'channels': 2048,
#     },
#     {
#         'type': 'deepgaze_pytorch.features.efficientnet.RGBEfficientNetB5',
#         'used_features': [
#             '1._blocks.24._depthwise_conv',
#             '1._blocks.26._depthwise_conv',
#             '1._blocks.35._project_conv',
#         ],
#         'channels': 2416,
#     },
#     {
#         'type': 'deepgaze_pytorch.features.densenet.RGBDenseNet201',
#         'used_features': [
#             '1.features.denseblock4.denselayer32.norm1',
#             '1.features.denseblock4.denselayer32.conv1',
#             '1.features.denseblock4.denselayer31.conv2',
#         ],
#         'channels': 2048,
#     },
#     {
#         'type': 'deepgaze_pytorch.features.resnext.RGBResNext50',
#         'used_features': [
#             '1.layer3.5.conv1',
#             '1.layer3.5.conv2',
#             '1.layer3.4.conv2',
#             '1.layer4.2.conv2',
#         ],
#         'channels': 2560,
#     },
# ]


# def build_saliency_network(input_channels):
#     return nn.Sequential(OrderedDict([
#         ('layernorm0', LayerNorm(input_channels)),
#         ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
#         ('bias0', Bias(8)),
#         ('softplus0', nn.Softplus()),

#         ('layernorm1', LayerNorm(8)),
#         ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
#         ('bias1', Bias(16)),
#         ('softplus1', nn.Softplus()),

#         ('layernorm2', LayerNorm(16)),
#         ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
#         ('bias2', Bias(1)),
#         ('softplus3', nn.Softplus()),
#     ]))


# def build_fixation_selection_network():
#     return nn.Sequential(OrderedDict([
#         ('layernorm0', LayerNormMultiInput([1, 0])),
#         ('conv0', Conv2dMultiInput([1, 0], 128, (1, 1), bias=False)),
#         ('bias0', Bias(128)),
#         ('softplus0', nn.Softplus()),

#         ('layernorm1', LayerNorm(128)),
#         ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
#         ('bias1', Bias(16)),
#         ('softplus1', nn.Softplus()),

#         ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
#     ]))


# def build_deepgaze_mixture(backbone_config, components=10):
#     feature_class = import_class(backbone_config['type'])
#     features = feature_class()

#     feature_extractor = FeatureExtractor(features, backbone_config['used_features'])

#     saliency_networks = []
#     scanpath_networks = []
#     fixation_selection_networks = []
#     finalizers = []
#     for component in range(components):
#         saliency_network = build_saliency_network(backbone_config['channels'])
#         fixation_selection_network = build_fixation_selection_network()

#         saliency_networks.append(saliency_network)
#         scanpath_networks.append(None)
#         fixation_selection_networks.append(fixation_selection_network)
#         finalizers.append(Finalizer(sigma=8.0, learn_sigma=True, saliency_map_factor=2))

#     return DeepGazeIIIMixture(
#         features=feature_extractor,
#         saliency_networks=saliency_networks,
#         scanpath_networks=scanpath_networks,
#         fixation_selection_networks=fixation_selection_networks,
#         finalizers=finalizers,
#         downsample=2,
#         readout_factor=16,
#         saliency_map_factor=2,
#         included_fixations=[],
#     )


# class DeepGazeIIE(MixtureModel):
#     """DeepGazeIIE model

#     :note
#     See Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021). Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling. ArXiv:2105.12441 [Cs], http://arxiv.org/abs/2105.12441
#     """
#     def __init__(self, pretrained=True):
#         # we average over 3 instances per backbone, each instance has 10 crossvalidation folds
#         backbone_models = [build_deepgaze_mixture(backbone_config, components=3 * 10) for backbone_config in BACKBONES]
#         super().__init__(backbone_models)

#         if pretrained:
#             self.load_state_dict(model_zoo.load_url('https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth', map_location=torch.device('cpu')))


# def import_class(name):
#     module_name, class_name = name.rsplit('.', 1)
#     module = importlib.import_module(module_name)
#     return getattr(module, class_name)

#deepgaze2e.py

from collections import OrderedDict
import importlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import numpy as np

from .modules import FeatureExtractor, Finalizer, DeepGazeIIIMixture, MixtureModel

from .layers import (
    Conv2dMultiInput,
    LayerNorm,
    LayerNormMultiInput,
    Bias,
)

BACKBONES = [
    {
        'type': 'deepgaze_pytorch.features.shapenet.RGBShapeNetC',
        'used_features': [
            '1.module.layer3.0.conv2',
            '1.module.layer3.3.conv2',
            '1.module.layer3.5.conv1',
            '1.module.layer3.5.conv2',
            '1.module.layer4.1.conv2',
            '1.module.layer4.2.conv2',
        ],
        'channels': 2048,
    },
    {
        'type': 'deepgaze_pytorch.features.efficientnet.RGBEfficientNetB5',
        'used_features': [
            '1._blocks.24._depthwise_conv',
            '1._blocks.26._depthwise_conv',
            '1._blocks.35._project_conv',
        ],
        'channels': 2416,
    },
    {
        'type': 'deepgaze_pytorch.features.densenet.RGBDenseNet201',
        'used_features': [
            '1.features.denseblock4.denselayer32.norm1',
            '1.features.denseblock4.denselayer32.conv1',
            '1.features.denseblock4.denselayer31.conv2',
        ],
        'channels': 2048,
    },
    {
        'type': 'deepgaze_pytorch.features.resnext.RGBResNext50',
        'used_features': [
            '1.layer3.5.conv1',
            '1.layer3.5.conv2',
            '1.layer3.4.conv2',
            '1.layer4.2.conv2',
        ],
        'channels': 2560,
    },
]

# def build_saliency_network(input_channels):
#     return nn.Sequential(OrderedDict([
#         ('layernorm0', LayerNorm(input_channels)),
#         ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
#         ('bias0', Bias(8)),
#         ('softplus0', nn.Softplus()),

#         ('layernorm1', LayerNorm(8)),
#         ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
#         ('bias1', Bias(16)),
#         ('softplus1', nn.Softplus()),

#         ('layernorm2', LayerNorm(16)),
#         ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
#         ('bias2', Bias(1)),
#         ('softplus3', nn.Softplus()),
#     ]))

def build_saliency_network(input_channels, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    # Crear las capas primero
    conv0 = nn.Conv2d(input_channels, 8, (1, 1), bias=False)
    conv1 = nn.Conv2d(8, 16, (1, 1), bias=False)
    conv2 = nn.Conv2d(16, 1, (1, 1), bias=False)
    
    # Inicializar los pesos
    nn.init.kaiming_normal_(conv0.weight)
    nn.init.kaiming_normal_(conv1.weight)
    nn.init.xavier_normal_(conv2.weight)
    
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', conv0),
        ('bias0', Bias(8)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(8)),
        ('conv1', conv1),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('layernorm2', LayerNorm(16)),
        ('conv2', conv2),
        ('bias2', Bias(1)),
        ('softplus3', nn.Softplus()),
    ]))


def build_fixation_selection_network():
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput([1, 0])),
        ('conv0', Conv2dMultiInput([1, 0], 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))

# def build_deepgaze_mixture(backbone_config, components=10): #, downsample=2, readout_factor=16, saliency_map_factor=2):
#     print(f"\nCreando mixture con {components} componentes")
#     feature_class = import_class(backbone_config['type'])
#     features = feature_class()
#     print(f"Backbone type: {backbone_config['type']}")

#     feature_extractor = FeatureExtractor(features, backbone_config['used_features'])

#     saliency_networks = []
#     scanpath_networks = []
#     fixation_selection_networks = []
#     finalizers = []
#     for component in range(components):
#         print(f"\nCreando componente {component + 1}/{components}")
#         saliency_network = build_saliency_network(backbone_config['channels'])
#         fixation_selection_network = build_fixation_selection_network()

#                 # Verificar pesos
#         total_params = sum(p.numel() for p in saliency_network.parameters())
#         sample_weights = next(saliency_network.parameters()).flatten()[:5]
#         print(f"Saliency Network {component} - Total params: {total_params}")
#         print(f"Sample weights: {sample_weights.tolist()}")

#         saliency_networks.append(saliency_network)
#         scanpath_networks.append(None)
#         fixation_selection_networks.append(fixation_selection_network)
#         # finalizers.append(Finalizer(sigma=8.0, learn_sigma=True, saliency_map_factor=saliency_map_factor))
#         finalizers.append(Finalizer(sigma=8.0, learn_sigma=True, saliency_map_factor=2))

#     return DeepGazeIIIMixture(
#         features=feature_extractor,
#         saliency_networks=saliency_networks,
#         scanpath_networks=scanpath_networks,
#         fixation_selection_networks=fixation_selection_networks,
#         finalizers=finalizers,
#         #downsample=2,
#         downsample=8.5,
#         readout_factor=16,
#         saliency_map_factor=2,
#         included_fixations=[],
#     )

def build_deepgaze_mixture(backbone_config, num_instances=3, num_folds=5, instance_seeds=None):
    print(f"\nCreando mixture para {backbone_config['type'].split('.')[-1]}")
    feature_class = import_class(backbone_config['type'])
    features = feature_class()

    feature_extractor = FeatureExtractor(features, backbone_config['used_features'])

    total_components = num_instances * num_folds
    print(f"Total de componentes (instancias × folds): {total_components}")

    saliency_networks = []
    scanpath_networks = []
    fixation_selection_networks = []
    finalizers = []
    
    for inst in range(num_instances):
        for fold in range(num_folds):
            seed = instance_seeds[inst][fold]
            print(f"\nCreando instancia {inst+1}, fold {fold+1} con seed {seed}")
            
            # Establecer seed para esta instancia/fold
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            saliency_network = build_saliency_network(backbone_config['channels'])
            
            # Verificar inicialización
            with torch.no_grad():
                for name, param in saliency_network.named_parameters():
                    if 'weight' in name:
                        print(f"\nPesos de {name}:")
                        print(f"Mean: {param.mean().item():.4f}")
                        print(f"Std: {param.std().item():.4f}")
            
            fixation_selection_network = build_fixation_selection_network()
            finalizer = Finalizer(sigma=8.0, learn_sigma=True, saliency_map_factor=2)
            
            saliency_networks.append(saliency_network)
            scanpath_networks.append(None)
            fixation_selection_networks.append(fixation_selection_network)
            finalizers.append(finalizer)

    return DeepGazeIIIMixture(
        features=feature_extractor,
        saliency_networks=saliency_networks,
        scanpath_networks=scanpath_networks,
        fixation_selection_networks=fixation_selection_networks,
        finalizers=finalizers,
        downsample=8.5,
        readout_factor=16,
        saliency_map_factor=2,
        included_fixations=[],
    )

# CLASIFICACION
class AdaptiveCategoryWeighting(nn.Module):
    def __init__(self, num_encoders=4, num_categories=20):
        super().__init__()
        self.num_encoders = num_encoders
        self.num_categories = num_categories
        
        # Matriz de pesos: [num_categories, num_encoders]
        self.weights = nn.Parameter(torch.ones(num_categories, num_encoders))
        self.softmax = nn.Softmax(dim=1)
        
        # Inicialización uniforme
        nn.init.constant_(self.weights, 1.0 / num_encoders)
    
    def forward(self, maps, category_indices):
        """
        Args:
            maps: [batch_size, num_encoders, height, width]
            category_indices: [batch_size]
        Returns:
            weighted_maps: [batch_size, 1, height, width]
        """
        # Obtener pesos normalizados para cada categoría
        weights = self.softmax(self.weights)  # [num_categories, num_encoders]
        
        # Seleccionar pesos para cada muestra en el batch
        batch_weights = weights[category_indices]  # [batch_size, num_encoders]
        
        # Expandir pesos para multiplicación
        weights_expanded = batch_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Aplicar pesos y sumar
        weighted_maps = (maps * weights_expanded).sum(dim=1, keepdim=True)
        
        return weighted_maps

    def get_category_weights(self):
        """Retorna los pesos normalizados por categoría para análisis"""
        return self.softmax(self.weights).detach()


class DeepGazeIIE(MixtureModel):
    """DeepGazeIIE model

    :note
    See Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021). Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling. ArXiv:2105.12441 [Cs], http://arxiv.org/abs/2105.12441
    """
    # def __init__(self, num_categories, pretrained=True):
    #     # we average over 3 instances per backbone, each instance has 5 crossvalidation folds
    #     backbone_models = [build_deepgaze_mixture(backbone_config, components=3 * 5) for backbone_config in BACKBONES]
    #     super().__init__(backbone_models)

    def __init__(self, num_categories, pretrained=True):
        print("\nInicializando DeepGazeIIE")
        print(f"Número de categorías: {num_categories}")
        print(f"Número de backbones: {len(BACKBONES)}")
        
        # Generar seeds aleatorias para cada instancia y fold
        num_instances = 3
        num_folds = 5
        instance_seeds = {}
        
        for backbone_idx, backbone_config in enumerate(BACKBONES):
            backbone_name = backbone_config['type'].split('.')[-1]
            instance_seeds[backbone_name] = [
                [np.random.randint(0, 10000) for _ in range(num_folds)]  # seeds para cada fold
                for _ in range(num_instances)  # para cada instancia
            ]
            print(f"\nSeeds para {backbone_name}:")
            for inst in range(num_instances):
                print(f"Instancia {inst + 1}: {instance_seeds[backbone_name][inst]}")
        
        backbone_models = []
        for backbone_config in BACKBONES:
            backbone_name = backbone_config['type'].split('.')[-1]
            model = build_deepgaze_mixture(
                backbone_config, 
                num_instances=num_instances,
                num_folds=num_folds,
                instance_seeds=instance_seeds[backbone_name]
            )
            backbone_models.append(model)
        
        super().__init__(backbone_models)

        # Módulo de ponderación por categoría # CLASIFICACION
        self.category_weighting = AdaptiveCategoryWeighting(
            num_encoders=len(BACKBONES),
            num_categories=num_categories
        )

        # if pretrained:
        #     self.load_state_dict(model_zoo.load_url('https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth', map_location=torch.device('cpu')))
    
        if pretrained:
            # Cargar pesos pre-entrenados
            state_dict = model_zoo.load_url(
                'https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth',
                map_location=torch.device('cpu')
            )
            # Intentar cargar los pesos, ignorando errores de tamaños no coincidentes
            try:
                self.load_state_dict(state_dict)
            except RuntimeError as e:
                print("WARNING: No se pudieron cargar algunos pesos pre-entrenados debido a diferencias en la arquitectura")
                print(f"Error: {str(e)}")
                print("Continuando con inicialización aleatoria para algunas capas...")

    # CLASIFICACION
    def forward(self, x, centerbias, category_indices=None):
        """
        Args:
            x: Imagen de entrada
            centerbias: Center bias
            category_indices: Índices de categoría para cada imagen
        """
        if category_indices is None:
            # Si no hay categorías, usar el comportamiento original
            return super().forward(x, centerbias)
        
        # Obtener predicciones individuales # CLASIFICACION
        predictions = []
        for model in self.models:
            pred = model(x, centerbias)
            # Asegurar que pred tenga la forma correcta [batch_size, 1, height, width]
            if len(pred.shape) == 3:
                pred = pred.unsqueeze(1)
            predictions.append(pred)
        
        # Concatenar en dimensión de modelos [batch_size, num_models, height, width]
        predictions = torch.cat(predictions, dim=1)
        
        # Aplicar ponderación por categoría
        weighted_prediction = self.category_weighting(predictions, category_indices)
        
        return weighted_prediction.squeeze(1)  # Asegurar forma correcta de salida
        
        # # Obtener predicciones de cada backbone
        # predictions = []
        # for model in self.models:
        #     pred = model(x, centerbias)
        #     predictions.append(pred)
        
        # # Stack predictions: [batch, num_models, H, W]
        # predictions = torch.stack(predictions, dim=1)
        
        # # Aplicar ponderación por categoría
        # weighted_prediction = self.category_weighting(predictions, category_indices)
        
        # return weighted_prediction


def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

    # return DeepGazeIIIMixture(
    #     features=feature_extractor,
    #     saliency_networks=saliency_networks,
    #     scanpath_networks=scanpath_networks,
    #     fixation_selection_networks=fixation_selection_networks,
    #     finalizers=finalizers,
    #     downsample=downsample,
    #     readout_factor=readout_factor,
    #     saliency_map_factor=saliency_map_factor,
    #     included_fixations=[],
    # )

# class DeepGazeIIE(MixtureModel):
#     """DeepGazeIIE model con optimizaciones de memoria
    
#     Args:
#         pretrained (bool): Si cargar pesos pre-entrenados
#         downsample (int): Factor de reducción de resolución para features
#         readout_factor (int): Factor de reducción para el readout
#         saliency_map_factor (int): Factor de reducción para el mapa de saliencia
#     """
#     def __init__(self, pretrained=True, downsample=2, readout_factor=16, saliency_map_factor=2):
#         # Build models with specified parameters
#         backbone_models = [
#             build_deepgaze_mixture(
#                 backbone_config, 
#                 components=3 * 10,
#                 downsample=downsample,
#                 readout_factor=readout_factor,
#                 saliency_map_factor=saliency_map_factor
#             ) 
#             for backbone_config in BACKBONES
#         ]
        
#         super().__init__(backbone_models)

#         if pretrained:
#             # Cargar pesos pre-entrenados
#             state_dict = model_zoo.load_url(
#                 'https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth',
#                 map_location=torch.device('cpu')
#             )
#             # Intentar cargar los pesos, ignorando errores de tamaños no coincidentes
#             try:
#                 self.load_state_dict(state_dict)
#             except RuntimeError as e:
#                 print("WARNING: No se pudieron cargar algunos pesos pre-entrenados debido a diferencias en la arquitectura")
#                 print(f"Error: {str(e)}")
#                 print("Continuando con inicialización aleatoria para algunas capas...")

# def import_class(name):
#     module_name, class_name = name.rsplit('.', 1)
#     module = importlib.import_module(module_name)
#     return getattr(module, class_name)


