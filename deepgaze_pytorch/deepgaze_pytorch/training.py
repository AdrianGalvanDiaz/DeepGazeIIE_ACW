# training.py
import os
from datetime import datetime
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from boltons.fileutils import atomic_save, mkdir_p
from torch.cuda.amp import autocast, GradScaler

# Importaciones actualizadas
from deepgaze_pytorch.deepgaze2e import DeepGazeIIE
from deepgaze_pytorch.metrics import log_likelihood, nss, auc

def eval_epoch(model, dataset, baseline_ll, device, metrics=None, log_per_category=True):
    """Evalúa el modelo usando las métricas especificadas"""
    model.eval()
    
    if metrics is None:
        metrics = ['LL', 'IG', 'NSS', 'AUC']
    
    metric_scores = {}
    category_metrics = defaultdict(lambda: defaultdict(list))
    batch_weights = []
    
    with torch.no_grad():
        pbar = tqdm(dataset)
        for i, batch in enumerate(pbar):
            image = batch['image'].to(device)
            centerbias = batch['centerbias'].to(device)
            fixation_mask = batch['fixation_mask'].to(device)
            weights = batch['weight'].to(device)
            category = batch['category'].to(device)
            
            log_density = model(image, centerbias, category)
            
            # Calcular métricas por muestra
            for metric_name in metrics:
                if metric_name == 'LL':
                    score = log_likelihood(log_density, fixation_mask, weights=weights, reduction='none')
                elif metric_name == 'NSS':
                    score = nss(log_density, fixation_mask, weights=weights, reduction='none')
                elif metric_name == 'AUC':
                    score = auc(log_density, fixation_mask, weights=weights, reduction='none')
                elif metric_name == 'IG':
                    ll = log_likelihood(log_density, fixation_mask, weights=weights, reduction='none')
                    score = ll - baseline_ll
                
                # Almacenar score promedio del batch
                metric_scores.setdefault(metric_name, []).append(score.mean().item())
                
                # Almacenar métricas por categoría
                if log_per_category:
                    for idx in range(len(category)):
                        cat = category[idx].item()
                        cat_score = score[idx].item()
                        category_metrics[cat][metric_name].append(cat_score)
            
            batch_weights.append(weights.detach().cpu().numpy().sum())
            
            # Actualizar barra de progreso
            current_scores = {k: np.mean(v) for k, v in metric_scores.items()}
            pbar.set_description(' '.join(f'{k}: {v:.4f}' for k, v in current_scores.items()))
    
    # Calcular métricas globales finales
    data = {k: np.average(v, weights=batch_weights) for k, v in metric_scores.items()}
    
    # Calcular y agregar métricas por categoría
    if log_per_category:
        category_averages = {}
        for cat in category_metrics:
            category_averages[cat] = {
                metric: np.mean(scores) 
                for metric, scores in category_metrics[cat].items()
            }
        data['per_category'] = category_averages
    
    print("\nResultados finales:")
    for k, v in data.items():
        if k != 'per_category':
            print(f"{k}: {v:.4f}")
    
    return data

# def train_epoch(model, dataset, optimizer, device, scaler, accumulation_steps=4):
#     """
#     Entrena el modelo por una época usando mixed precision y gradient accumulation,
#     ahora con soporte para categorías
#     """
#     model.train()
#     losses = []
#     category_losses = defaultdict(list)
#     batch_weights = []
#     optimizer.zero_grad()  # Asegurarse de empezar limpios
    
#     pbar = tqdm(dataset)
#     for i, batch in enumerate(pbar):
#         # Mover datos a GPU
#         image = batch['image'].to(device)
#         centerbias = batch['centerbias'].to(device)
#         fixation_mask = batch['fixation_mask'].to(device)
#         weights = batch['weight'].to(device)
#         category = batch['category'].to(device)
        
#         # Mixed Precision Forward Pass
#         with autocast():
#             log_density = model(image, centerbias, category)
#             # Calcular pérdida por cada muestra en el batch
#             sample_losses = -log_likelihood(log_density, fixation_mask, weights=weights, reduction='none')
#             # Pérdida promedio del batch
#             loss = sample_losses.mean() / accumulation_steps
        
#         # Tracking pérdida por categoría
#         with torch.no_grad():
#             for idx in range(len(category)):
#                 cat = category[idx].item()
#                 cat_loss = sample_losses[idx].item()
#                 category_losses[cat].append(cat_loss)
        
#         # Mixed Precision Backward Pass
#         scaler.scale(loss).backward()
        
#         if (i + 1) % accumulation_steps == 0:
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
        
#         # Tracking de pérdida global
#         losses.append(loss.item() * accumulation_steps)
#         batch_weights.append(weights.sum().item())
        
#         # Actualizar barra de progreso
#         avg_loss = np.mean(losses)
#         pbar.set_description(f'Loss: {avg_loss:.05f}')
    
#     # Asegurarse de aplicar el último gradiente acumulado
#     if (i + 1) % accumulation_steps != 0:
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad()
    
#     # Calcular pérdidas promedio por categoría
#     category_avg_losses = {
#         cat: np.mean(losses) for cat, losses in category_losses.items()
#     }
    
#     # Retornar pérdida global y pérdidas por categoría
#     return np.mean(losses), category_avg_losses

def train_epoch(model, dataset, optimizer, device, scaler, accumulation_steps=4):
    """
    Entrena el modelo por una época usando mixed precision y gradient accumulation,
    ahora con soporte para categorías
    """
    model.train()
    losses = []
    category_losses = defaultdict(list)
    batch_weights = []
    optimizer.zero_grad()  # Asegurarse de empezar limpios
    
    pbar = tqdm(dataset)
    for i, batch in enumerate(pbar):
        # Mover datos a GPU
        image = batch['image'].to(device)
        centerbias = batch['centerbias'].to(device)
        fixation_mask = batch['fixation_mask'].to(device)
        weights = batch['weight'].to(device)
        category = batch['category'].to(device)
        
        # Mixed Precision Forward Pass
        with autocast():
            log_density = model(image, centerbias, category)
            # Calcular pérdida por cada muestra en el batch
            sample_losses = -log_likelihood(log_density, fixation_mask, weights=weights, reduction='none')
            # Pérdida promedio del batch
            loss = sample_losses.mean() / accumulation_steps
        
        # Tracking pérdida por categoría
        with torch.no_grad():
            for idx in range(len(category)):
                cat = category[idx].item()
                cat_loss = sample_losses[idx].item()
                category_losses[cat].append(cat_loss)
        
        # Mixed Precision Backward Pass
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Tracking de pérdida global
        losses.append(loss.item() * accumulation_steps)
        batch_weights.append(weights.sum().item())
        
        # Actualizar barra de progreso
        avg_loss = np.mean(losses)
        pbar.set_description(f'Loss: {avg_loss:.05f}')
    
    # Asegurarse de aplicar el último gradiente acumulado
    if (i + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Calcular pérdidas promedio por categoría
    category_avg_losses = {
        cat: np.mean(losses) for cat, losses in category_losses.items()
    }
    
    # Retornar pérdida global y pérdidas por categoría
    return np.mean(losses), category_avg_losses

def save_checkpoint(model, optimizer, scheduler, scaler, step, loss, directory, category_losses=None):
    """Guarda el estado del entrenamiento y métricas"""
    data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'step': step,
        'loss': loss,
    }
    
    # Guardar pérdidas por categoría si están disponibles
    if category_losses is not None:
        data['category_losses'] = category_losses
    
    checkpoint_path = os.path.join(directory, f'step-{step:04d}.pth')
    with atomic_save(checkpoint_path, text_mode=False, overwrite_part=True) as f:
        torch.save(data, f)
    return checkpoint_path

def train(model, train_loader, val_loader, train_baseline_ll, val_baseline_ll, optimizer, 
          scheduler, checkpoint_dir, device=None, validation_epochs=1, 
          validation_metrics=['IG', 'LL', 'AUC', 'NSS'], 
          minimum_learning_rate=1e-6, accumulation_steps=4):
    """
    Función principal de entrenamiento con soporte para categorías y logging extendido
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scaler = GradScaler()
    mkdir_p(checkpoint_dir)
    
    # Configurar TensorBoard
    tensorboard_dir = os.path.join(checkpoint_dir, 'logs')
    writer = SummaryWriter(tensorboard_dir, flush_secs=30)
    
    # Preparar DataFrame para tracking con columnas adicionales
    base_columns = ['epoch', 'timestamp', 'learning_rate', 'loss']
    metric_columns = [f'validation_{m}' for m in validation_metrics]
    category_columns = []
    
    # Agregar columnas para pérdidas por categoría
    categories = train_loader.dataset.category_to_idx
    for cat_name in categories:
        category_columns.extend([
            f'train_loss_{cat_name}',
            f'val_ig_{cat_name}',
            f'val_ll_{cat_name}'
        ])
    
    columns = base_columns + metric_columns + category_columns
    progress = pd.DataFrame(columns=columns)

    model.to(device)
    step = 0
    
    # Cargar checkpoint si existe
    existing_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'step-*.pth')))
    if existing_checkpoints:
        print(f"Resumiendo entrenamiento desde {existing_checkpoints[-1]}")
        checkpoint = torch.load(existing_checkpoints[-1])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        step = checkpoint['step']
    
    while optimizer.param_groups[0]['lr'] >= minimum_learning_rate:
        step += 1
        print(f"\nEpoch {step}")
        
        # Entrenamiento
        loss, category_losses = train_epoch(
            model, train_loader, optimizer, device, scaler, accumulation_steps
        )
        
        # Logging global
        writer.add_scalar('training/loss', loss, step)
        writer.add_scalar('training/learning_rate', optimizer.param_groups[0]['lr'], step)
        
        # Logging por categoría
        for cat, cat_loss in category_losses.items():
            writer.add_scalar(f'training/category_{cat}_loss', cat_loss, step)
        
        # Validación
        if step % validation_epochs == 0:
            print("\nRunning validation...")
            val_data = eval_epoch(
                model, val_loader, val_baseline_ll, device, 
                metrics=validation_metrics, log_per_category=True
            )
            
            # Logging métricas globales
            for metric, value in val_data.items():
                if metric != 'per_category':
                    writer.add_scalar(f'validation/{metric}', value, step)
            
            # Logging métricas por categoría
            if 'per_category' in val_data:
                for cat, metrics in val_data['per_category'].items():
                    for metric, value in metrics.items():
                        writer.add_scalar(f'validation/category_{cat}_{metric}', value, step)

            # Guardar análisis de pesos por categoría
            if hasattr(model, 'category_weighting'):
                weights = model.category_weighting.get_category_weights()
                np.save(
                    os.path.join(checkpoint_dir, f'category_weights_epoch_{step}.npy'),
                    weights.cpu().numpy()
                )
            
            # Actualizar scheduler
            if 'IG' in val_data:
                scheduler.step(val_data['IG'])
            else:
                scheduler.step(val_data.get('LL', 0))


        # Actualizar progreso
        new_row = {
            'epoch': step,
            'timestamp': datetime.utcnow(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'loss': loss
        }
        
        # Agregar métricas de validación globales
        for metric, value in val_data.items():
            if metric != 'per_category':
                new_row[f'validation_{metric}'] = value
    
        # Agregar métricas por categoría
        if 'per_category' in val_data:
            for cat_name, cat_idx in categories.items():
            # Pérdida de entrenamiento por categoría
                if cat_idx in category_losses:
                    new_row[f'train_loss_{cat_name}'] = category_losses[cat_idx]
            
            # Métricas de validación por categoría
                if cat_idx in val_data['per_category']:
                    cat_metrics = val_data['per_category'][cat_idx]
                    new_row[f'val_ig_{cat_name}'] = cat_metrics.get('IG', np.nan)
                    new_row[f'val_ll_{cat_name}'] = cat_metrics.get('LL', np.nan)
    
        progress.loc[step] = new_row

        # Guardar checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            loss=loss,
            category_losses=category_losses,  # Nuevo
            directory=checkpoint_dir
        )
        
        # Guardar el CSV con toda la información
        progress.to_csv(os.path.join(checkpoint_dir, 'training_log.csv'))
    
    # Guardar modelo final
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final.pth'))
    print("\nTraining completed. Final model saved.")