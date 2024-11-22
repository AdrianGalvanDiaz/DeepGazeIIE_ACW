#modules.py

import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GaussianFilterNd


def encode_scanpath_features(x_hist, y_hist, size, device=None, include_x=True, include_y=True, include_duration=False):
    assert include_x
    assert include_y
    assert not include_duration

    height = size[0]
    width = size[1]

    xs = torch.arange(width, dtype=torch.float32).to(device)
    ys = torch.arange(height, dtype=torch.float32).to(device)
    YS, XS = torch.meshgrid(ys, xs, indexing='ij')

    XS = torch.repeat_interleave(
        torch.repeat_interleave(
            XS[np.newaxis, np.newaxis, :, :],
            repeats=x_hist.shape[0],
            dim=0,
        ),
        repeats=x_hist.shape[1],
        dim=1,
    )

    YS = torch.repeat_interleave(
        torch.repeat_interleave(
            YS[np.newaxis, np.newaxis, :, :],
            repeats=y_hist.shape[0],
            dim=0,
        ),
        repeats=y_hist.shape[1],
        dim=1,
    )

    XS -= x_hist.unsqueeze(2).unsqueeze(3)
    YS -= y_hist.unsqueeze(2).unsqueeze(3)

    distances = torch.sqrt(XS**2 + YS**2)

    return torch.cat((XS, YS, distances), axis=1)

class FeatureExtractor(torch.nn.Module):
    def __init__(self, features, targets):
        super().__init__()
        self.features = features
        self.targets = targets
        #print("Targets are {}".format(targets))
        self.outputs = {}

        for target in targets:
            layer = dict([*self.features.named_modules()])[target]
            layer.register_forward_hook(self.save_outputs_hook(target))

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.outputs[layer_id] = output.clone()
        return fn

    def forward(self, x):

        self.outputs.clear()
        self.features(x)
        return [self.outputs[target] for target in self.targets]


def upscale(tensor, size):
    tensor_size = torch.tensor(tensor.shape[2:]).type(torch.float32)
    target_size = torch.tensor(size).type(torch.float32)
    factors = torch.ceil(target_size / tensor_size)
    factor = torch.max(factors).type(torch.int64).to(tensor.device)
    assert factor >= 1

    tensor = torch.repeat_interleave(tensor, factor, dim=2)
    tensor = torch.repeat_interleave(tensor, factor, dim=3)

    tensor = tensor[:, :, :size[0], :size[1]]

    return tensor

class Finalizer(nn.Module):
    def __init__(
        self,
        sigma,
        kernel_size=None,
        learn_sigma=False,
        center_bias_weight=1.0,
        learn_center_bias_weight=True,
        saliency_map_factor=4,
        intermediate_scales=[2, 4, 8]
    ):
        super(Finalizer, self).__init__()
        self.saliency_map_factor = saliency_map_factor
        self.intermediate_scales = intermediate_scales
        
        self.gaussian_filters = nn.ModuleList([
            GaussianFilterNd([2, 3], sigma/scale, truncate=3, trainable=learn_sigma)
            for scale in intermediate_scales
        ])
        self.center_bias_weight = nn.Parameter(torch.Tensor([center_bias_weight]), requires_grad=learn_center_bias_weight)

    def forward(self, readout, centerbias):
        # print("\n=== Finalizer Debug Inicio ===")
        # print(f"Input readout shape: {readout.shape}")
        # print(f"Input centerbias shape: {centerbias.shape}")
        # print(f"Initial readout range: [{readout.min().item():.4f}, {readout.max().item():.4f}]")
        
        # Escalamiento progresivo
        out = readout
        current_size = [out.shape[2], out.shape[3]]
        target_size = [centerbias.shape[1], centerbias.shape[2]]
        
        # print("\n=== Escalamiento Progresivo ===")
        for i, scale in enumerate(self.intermediate_scales):
            # print(f"\nEtapa de escalamiento {i+1}:")
            # print(f"Factor de escala: {scale}")
            
            new_size = [
                min(current_size[0] * scale, target_size[0]),
                min(current_size[1] * scale, target_size[1])
            ]
            
            out = F.interpolate(
                out,
                size=new_size,
                mode='bicubic',
                align_corners=True
            )
            # print(f"Shape después de interpolación: {out.shape}")
            # print(f"Rango después de interpolación: [{out.min().item():.4f}, {out.max().item():.4f}]")
            
            # out = self.gaussian_filters[i](out)
            # print(f"Sigma del filtro gaussiano: {self.gaussian_filters[i].sigma.item() if hasattr(self.gaussian_filters[i], 'sigma') else 'N/A'}")
            # print(f"Rango después de gaussiano: [{out.min().item():.4f}, {out.max().item():.4f}]")
            
            current_size = new_size
        
        # Interpolación final
        if current_size != target_size:
            # print("\n=== Interpolación Final ===")
            # print(f"Escalando de {current_size} a {target_size}")
            out = F.interpolate(
                out,
                size=target_size,
                mode='bicubic',
                align_corners=True
            )
            # print(f"Shape final: {out.shape}")
            # print(f"Rango después de interpolación final: [{out.min().item():.4f}, {out.max().item():.4f}]")
        
        # Remover dimensión de canal y preparar para suma con center bias
        out = out[:, 0, :, :]
        
        # Escalar center bias al tamaño final
        # print("\n=== Center Bias Processing ===")
        centerbias_full = F.interpolate(
            centerbias.unsqueeze(1),
            size=target_size,
            mode='bilinear',
            align_corners=True
        )[:, 0, :, :]
        # print(f"Center bias final shape: {centerbias_full.shape}")
        # print(f"Center bias range: [{centerbias_full.min().item():.4f}, {centerbias_full.max().item():.4f}]")
        
        # Añadir center bias
        out = out + self.center_bias_weight * centerbias_full
        # print(f"\n=== Procesamiento Final ===")
        # print(f"Center bias weight: {self.center_bias_weight.item():.4f}")
        # print(f"Rango después de añadir center bias: [{out.min().item():.4f}, {out.max().item():.4f}]")
        
        # Normalización final
        out = out - torch.logsumexp(out.reshape(out.shape[0], -1), dim=1).reshape(-1, 1, 1)
        
        # print("\n=== Resultados Finales ===")
        # print(f"Shape final: {out.shape}")
        # print(f"Rango final: [{out.min().item():.4f}, {out.max().item():.4f}]")
        # print("===========================")
        
        return out

# class Finalizer(nn.Module):
#     """Transforms a readout into a gaze prediction

#     A readout network returns a single, spatial map of probable gaze locations.
#     This module bundles the common processing steps necessary to transform this into
#     the predicted gaze distribution:

#      - resizing to the stimulus size
#      - smoothing of the prediction using a gaussian filter
#      - removing of channel and time dimension
#      - weighted addition of the center bias
#      - normalization
#     """

#     def __init__(
#         self,
#         sigma,
#         kernel_size=None,
#         learn_sigma=False,
#         center_bias_weight=1.0,
#         learn_center_bias_weight=True,
#         saliency_map_factor=4,
#     ):
#         """Creates a new finalizer

#         Args:
#             size (tuple): target size for the predictions
#             sigma (float): standard deviation of the gaussian kernel used for smoothing
#             kernel_size (int, optional): size of the gaussian kernel
#             learn_sigma (bool, optional): If True, the standard deviation of the gaussian kernel will
#                 be learned (default: False)
#             center_bias (string or tensor): the center bias
#             center_bias_weight (float, optional): initial weight of the center bias
#             learn_center_bias_weight (bool, optional): If True, the center bias weight will be
#                 learned (default: True)
#         """
#         super(Finalizer, self).__init__()

#         self.saliency_map_factor = saliency_map_factor

#         self.gauss = GaussianFilterNd([2, 3], sigma, truncate=3, trainable=learn_sigma)
#         self.center_bias_weight = nn.Parameter(torch.Tensor([center_bias_weight]), requires_grad=learn_center_bias_weight)

#     def forward(self, readout, centerbias):
#         """Applies the finalization steps to the given readout"""

#         # print("\nFinalizer forward:")
#         # print(f"Center bias weight: {self.center_bias_weight.item():.4f}")
#         # print(f"Input range: [{readout.min().item():.4f}, {readout.max().item():.4f}]")

#         downscaled_centerbias = F.interpolate(
#             centerbias.view(centerbias.shape[0], 1, centerbias.shape[1], centerbias.shape[2]),
#             scale_factor=1 / self.saliency_map_factor,
#             recompute_scale_factor=False,
#         )[:, 0, :, :]

#         out = F.interpolate(
#             readout,
#             size=[downscaled_centerbias.shape[1], downscaled_centerbias.shape[2]]
#         )

#         # apply gaussian filter
#         out = self.gauss(out)

#         # remove channel dimension
#         out = out[:, 0, :, :]

#         # add to center bias
#         out = out + self.center_bias_weight * downscaled_centerbias

#         out = F.interpolate(out[:, np.newaxis, :, :], size=[centerbias.shape[1], centerbias.shape[2]])[:, 0, :, :]

#         out = out - torch.logsumexp(out.reshape(out.shape[0], -1), dim=1).reshape(-1, 1, 1)

#         return out


class DeepGazeII(torch.nn.Module):
    def __init__(self, features, readout_network, downsample=8.5, readout_factor=16, saliency_map_factor=2, initial_sigma=8.0):
        super().__init__()

        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.readout_network = readout_network
        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=True,
            saliency_map_factor=self.saliency_map_factor,
        )
        self.downsample = downsample

    def forward(self, x, centerbias):
        orig_shape = x.shape
        x = F.interpolate(
            x,
            scale_factor=1 / self.downsample,
            recompute_scale_factor=False,
        )
        x = self.features(x)

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [F.interpolate(item, readout_shape) for item in x]

        x = torch.cat(x, dim=1)
        x = self.readout_network(x)
        x = self.finalizer(x, centerbias)

        return x

    def train(self, mode=True):
        self.features.eval()
        self.readout_network.train(mode=mode)
        self.finalizer.train(mode=mode)


class DeepGazeIII(torch.nn.Module):
    def __init__(self, features, saliency_network, scanpath_network, fixation_selection_network, downsample=8.5, readout_factor=2, saliency_map_factor=2, included_fixations=-2, initial_sigma=8.0):
        super().__init__()

        self.downsample = downsample
        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor
        self.included_fixations = included_fixations

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network

        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=True,
            saliency_map_factor=self.saliency_map_factor,
        )

    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None):
        orig_shape = x.shape
        x = F.interpolate(x, scale_factor=1 / self.downsample)
        x = self.features(x)

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [F.interpolate(item, readout_shape) for item in x]

        x = torch.cat(x, dim=1)
        x = self.saliency_network(x)

        if self.scanpath_network is not None:
            scanpath_features = encode_scanpath_features(x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=x.device)
            #scanpath_features = F.interpolate(scanpath_features, scale_factor=1 / self.downsample / self.readout_factor)
            scanpath_features = F.interpolate(scanpath_features, readout_shape)
            y = self.scanpath_network(scanpath_features)
        else:
            y = None

        x = self.fixation_selection_network((x, y))

        x = self.finalizer(x, centerbias)

        return x

    def train(self, mode=True):
        self.features.eval()
        self.saliency_network.train(mode=mode)
        if self.scanpath_network is not None:
            self.scanpath_network.train(mode=mode)
        self.fixation_selection_network.train(mode=mode)
        self.finalizer.train(mode=mode)


class DeepGazeIIIMixture(torch.nn.Module):
    def __init__(self, features, saliency_networks, scanpath_networks, fixation_selection_networks, finalizers, downsample=8.5, readout_factor=2, saliency_map_factor=2, included_fixations=-2, initial_sigma=8.0):
        super().__init__()

        self.downsample = downsample
        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor
        self.included_fixations = included_fixations

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.saliency_networks = torch.nn.ModuleList(saliency_networks)
        self.scanpath_networks = torch.nn.ModuleList(scanpath_networks)
        self.fixation_selection_networks = torch.nn.ModuleList(fixation_selection_networks)
        self.finalizers = torch.nn.ModuleList(finalizers)

    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None):
        # print("\nForward pass de DeepGazeIIIMixture")
        # print(f"Número de redes: {len(self.saliency_networks)}")
        orig_shape = x.shape
        x = F.interpolate(
            x,
            scale_factor=1 / self.downsample,
            recompute_scale_factor=False,
        )
        x = self.features(x)

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [F.interpolate(item, readout_shape) for item in x]

        x = torch.cat(x, dim=1)

        predictions = []

        readout_input = x

        # print(f"Número de redes en este backbone: {len(self.saliency_networks)}")
        count = 0
        for saliency_network, scanpath_network, fixation_selection_network, finalizer in zip(
            self.saliency_networks, self.scanpath_networks, self.fixation_selection_networks, self.finalizers
        ):

            count += 1
            # print(f"Generando mapa {count}")

            x = saliency_network(readout_input)

            if scanpath_network is not None:
                scanpath_features = encode_scanpath_features(x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=x.device)
                scanpath_features = F.interpolate(scanpath_features, readout_shape)
                y = scanpath_network(scanpath_features)
            else:
                y = None

            x = fixation_selection_network((x, y))

            x = finalizer(x, centerbias)

            predictions.append(x[:, np.newaxis, :, :])

        # print(f"Total mapas generados: {count}")

        predictions = torch.cat(predictions, dim=1) - np.log(len(self.saliency_networks))

        prediction = predictions.logsumexp(dim=(1), keepdim=True)

        return prediction

# CLASIFICACION
class MixtureModel(torch.nn.Module):
    """Versión modificada del MixtureModel para soportar ponderación por categoría"""
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x, centerbias, category_indices=None):
        """
        Args:

            x: Tensor de entrada [batch_size, channels, height, width]
            centerbias: Tensor del center bias [batch_size, height, width]
            category_indices: Tensor opcional de índices de categoría [batch_size]
        """
        # Obtener predicciones individuales de cada modelo
        predictions = [model.forward(x, centerbias) for model in self.models]
        
        # Stack predictions para formato [batch_size, num_models, height, width]
        predictions = torch.stack(predictions, dim=1)
        
        # Si no hay categorías, usar el promedio simple (comportamiento original)
        if category_indices is None:
            predictions = predictions - np.log(len(self.models))
            prediction = predictions.logsumexp(dim=1)
            return prediction[:, None, :, :]  # Agregar dim de canal
            
        # La ponderación por categoría se maneja en DeepGazeIIE
        return predictions

    def get_individual_predictions(self, x, centerbias):
        """
        Método de utilidad para obtener predicciones individuales de cada modelo
        Útil para análisis y visualización
        """
        with torch.no_grad():
            predictions = [model.forward(x, centerbias) for model in self.models]
            return torch.stack(predictions, dim=1)

# FUNCIONA GAUSS
# class MixtureModel(torch.nn.Module):
#     def __init__(self, models):
#         super().__init__()
#         self.models = torch.nn.ModuleList(models)

#     def forward(self, *args, **kwargs):
#         predictions = [model.forward(*args, **kwargs) for model in self.models]
        
#         # # Debug prints
#         # print("Antes de combinar modelos:")
#         # for i, pred in enumerate(predictions):
#         #     print(f"Modelo {i}:")
#         #     print(f"  Min: {pred.min().item():.4f}")
#         #     print(f"  Max: {pred.max().item():.4f}")
#         #     print(f"  Mean: {pred.mean().item():.4f}")
#         #     print(f"  exp(pred).sum(): {torch.exp(pred).sum().item():.4f}")

#         predictions = torch.cat(predictions, dim=1)
#         predictions -= np.log(len(self.models))
#         prediction = predictions.logsumexp(dim=(1), keepdim=True)

#         # print("Después de combinar:")
#         # print(f"Min: {prediction.min().item():.4f}")
#         # print(f"Max: {prediction.max().item():.4f}")
#         # print(f"Mean: {prediction.mean().item():.4f}")
#         # print(f"exp(prediction).sum(): {torch.exp(prediction).sum().item():.4f}")

#         return prediction