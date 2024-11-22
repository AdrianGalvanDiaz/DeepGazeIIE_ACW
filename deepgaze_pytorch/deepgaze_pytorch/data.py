#data.py

from collections import Counter
import os
import random
import numpy as np
import torch
import scipy.io
from PIL import Image
from torch.utils.data import Dataset, random_split, Subset
import pysaliency
from pysaliency import FileStimuli, Fixations
from tqdm import tqdm
import numpy as np
from pysaliency import Model, FileStimuli, Fixations

def load_cat2000_data(root_dir):
    """
    Carga los datos de CAT2000 y retorna también el mapeo de categorías
    """
    stimuli_dir = os.path.join(root_dir, 'CAT2000', 'trainSet', 'Stimuli')
    fixations_dir = os.path.join(root_dir, 'CAT2000', 'trainSet', 'FIXATIONLOCS')
    
    image_paths = []
    fixation_paths = []
    categories = []
    category_to_idx = {}
    
    # Obtener lista de categorías
    category_dirs = [d for d in os.listdir(stimuli_dir)
                    if os.path.isdir(os.path.join(stimuli_dir, d))]
    
    print(f"Categorías encontradas: {len(category_dirs)}")
    for cat in sorted(category_dirs):
        category_to_idx[cat] = len(category_to_idx)
    
    for category in category_dirs:
        category_img_dir = os.path.join(stimuli_dir, category)
        category_fix_dir = os.path.join(fixations_dir, category)
        
        img_files = [f for f in os.listdir(category_img_dir)
                    if f.endswith('.jpg')]
        
        print(f"Categoría {category}: {len(img_files)} imágenes")
        
        for img_file in img_files:
            mat_file = os.path.splitext(img_file)[0] + '.mat'
            img_path = os.path.join(category_img_dir, img_file)
            mat_path = os.path.join(category_fix_dir, mat_file)
            
            if os.path.exists(mat_path):
                image_paths.append(img_path)
                fixation_paths.append(mat_path)
                categories.append(category_to_idx[category])
    
    print(f"\nEstadísticas del dataset:")
    print(f"Total de imágenes: {len(image_paths)}")
    print("\nDistribución por categoría:")
    cat_counts = {cat: categories.count(idx) for cat, idx in category_to_idx.items()}
    for cat, count in cat_counts.items():
        print(f"{cat}: {count} imágenes")
    
    return image_paths, fixation_paths, categories, category_to_idx

class CenterBiasModel(Model):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self._cached_log_density = None
        self._cached_shape = None

    def _log_density(self, stimulus):
        if stimulus is None:
            raise ValueError("Stimulus cannot be None")
            
        height, width = stimulus.shape[:2]
        
        if self._cached_log_density is not None and self._cached_shape == (height, width):
            return self._cached_log_density
        
        # Debug pre-computación
        print("\nGenerando nuevo center bias:")
        print(f"Shape de entrada: {(height, width)}")
        
        # Crear coordenadas normalizadas
        y = np.linspace(-1, 1, height, dtype=np.float32)[:, np.newaxis]
        x = np.linspace(-1, 1, width, dtype=np.float32)[np.newaxis, :]
        
        # Calcular gaussiana
        gaussian = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        
        # Debug pre-normalización
        print(f"Gaussian stats antes de normalizar:")
        print(f"Min: {gaussian.min():.4f}")
        print(f"Max: {gaussian.max():.4f}")
        print(f"Sum: {gaussian.sum():.4f}")
        
        # Normalización exacta como en el paper
        gaussian = gaussian / gaussian.sum()
        log_density = np.log(gaussian + 1e-20)  # mismo epsilon que el paper
        log_density = log_density - scipy.special.logsumexp(log_density)
        
        # Debug post-normalización
        print(f"\nCenter Bias final stats:")
        print(f"Min value (log): {log_density.min():.4f}")
        print(f"Max value (log): {log_density.max():.4f}")
        print(f"Sum exp(log_density): {np.exp(log_density).sum():.4f}")
        
        self._cached_log_density = log_density
        self._cached_shape = (height, width)
        
        return log_density

def create_center_bias_model(stimuli, fixations, verbose=True):
    print("Creando center bias model...")
    centerbias_model = CenterBiasModel()
    
    if verbose:
        total_ll = 0.0
        count = 0
        
        for n in tqdm(range(len(stimuli)), desc="Calculando baseline"):
            stimulus = stimuli.stimuli[n]
            fix_inds = fixations.n == n
            
            if not np.any(fix_inds):
                continue
            
            x = fixations.x[fix_inds]
            y = fixations.y[fix_inds]
            
            log_density = centerbias_model.log_density(stimulus)
            current_ll = np.sum(log_density[y.astype(int), x.astype(int)])
            
            if len(x) > 0:
                total_ll += current_ll
                count += len(x)
                
                if n % 100 == 0:
                    print(f"\nImagen {n}:")
                    print(f"Log-likelihood promedio (nats): {current_ll/len(x):.4f}")
                    print(f"Número de fijaciones: {len(x)}")
        
        baseline_ll = total_ll / count
        
        print(f"\nBaseline Log-Likelihood (nats): {baseline_ll:.4f}")
        
        return centerbias_model, baseline_ll
    
class FixationMaskTransform(object):
    """
    Transforma coordenadas de fijación en una máscara binaria.
    """
    def __init__(self, sparse=True):
        super().__init__()
        self.sparse = sparse

    def __call__(self, item):
        shape = torch.Size([item['image'].shape[1], item['image'].shape[2]])
        x = item.pop('x')
        y = item.pop('y')

        inds = np.array([y, x])
        values = np.ones(len(y), dtype=int)

        mask = torch.sparse.IntTensor(torch.tensor(inds), torch.tensor(values), shape)
        mask = mask.coalesce()
        
        if not self.sparse:
            mask = mask.to_dense()

        item['fixation_mask'] = mask
        return item

class CAT2000Dataset(Dataset):
    def __init__(self, image_paths, fixation_paths, categories, category_to_idx, 
                 centerbias_model=None, transform=None):
        self.image_paths = image_paths
        self.fixation_paths = fixation_paths
        self.categories = categories
        self.category_to_idx = category_to_idx
        self.centerbias_model = centerbias_model
        self.transform = transform
        
        print(f"\nInicializando dataset:")
        print(f"Total imágenes: {len(image_paths)}")
        print(f"Total categorías: {len(category_to_idx)}")
        self._validate_data()
    
    def _validate_data(self):
        """Valida la integridad de los datos"""
        assert len(self.image_paths) == len(self.fixation_paths) == len(self.categories), \
            "Mismatch en longitudes de datos"
        
        # Verificar que todas las categorías están presentes
        unique_cats = set(self.categories)
        expected_cats = set(range(len(self.category_to_idx)))
        assert unique_cats == expected_cats, \
            f"Categorías faltantes: {expected_cats - unique_cats}"

    def __getitem__(self, idx):
        try:
            # Cargar imagen
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image_np = np.array(image)
            
            # Calcular center bias
            if self.centerbias_model is not None:
                centerbias = self.centerbias_model.log_density(image_np)
                centerbias = torch.FloatTensor(centerbias)
            else:
                centerbias = torch.ones(image_np.shape[:2])
            
            # Convertir imagen para PyTorch
            image = np.array(image).astype(np.float32)
            image = image.transpose(2, 0, 1)
            image = torch.FloatTensor(image)
            
            # Cargar fijaciones
            mat_data = scipy.io.loadmat(self.fixation_paths[idx])
            fixation_map = mat_data['fixLocs']
            y_coords, x_coords = np.where(fixation_map > 0)
            
            data = {
                'image': image,
                'x': torch.LongTensor(x_coords),
                'y': torch.LongTensor(y_coords),
                'centerbias': centerbias,
                'category': torch.tensor(self.categories[idx], dtype=torch.long),
                'category_name': list(self.category_to_idx.keys())[list(self.category_to_idx.values()).index(self.categories[idx])],
                'weight': torch.tensor(1.0),
            }
            
            if self.transform:
                data = self.transform(data)
            
            return data
            
        except Exception as e:
            print(f"Error en __getitem__ para idx {idx}: {str(e)}")
            print(f"Imagen: {self.image_paths[idx]}")
            print(f"Categoría: {self.categories[idx]}")
            raise e

    def __len__(self):
        return len(self.image_paths)

class CAT2000SplitDataset(Subset):
    """Wrapper para mantener acceso a los paths y categorías después del split"""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.image_paths = [dataset.image_paths[i] for i in indices]
        self.fixation_paths = [dataset.fixation_paths[i] for i in indices]
        self.categories = [dataset.categories[i] for i in indices]
        self.category_to_idx = dataset.category_to_idx

def create_pysaliency_datasets(image_paths, fixation_paths, scale_factor=8.5):
    """
    Crea los datasets de pysaliency con escalado correcto.
    """
    print("Creando datasets de pysaliency...")
    
    # Crear stimuli
    stimuli = FileStimuli(image_paths)
    
    all_x = []
    all_y = []
    all_n = []
    
    for n, fix_path in enumerate(tqdm(fixation_paths, desc="Procesando fijaciones")):
        fix_matrix = scipy.io.loadmat(fix_path)['fixLocs']
        y_coords, x_coords = np.nonzero(fix_matrix)
        
        # Escalar coordenadas
        x_coords = (x_coords / scale_factor).astype(int)
        y_coords = (y_coords / scale_factor).astype(int)
        
        all_x.extend(x_coords)
        all_y.extend(y_coords)
        all_n.extend([n] * len(x_coords))
    
    print(f"\nEstadísticas después de escalar:")
    print(f"Rango x: {min(all_x)} - {max(all_x)}")
    print(f"Rango y: {min(all_y)} - {max(all_y)}")
    
    fixations = Fixations(
        x=all_x,
        y=all_y,
        n=all_n,
        subjects=np.zeros_like(all_n),
        t=np.zeros(len(all_x)),
        x_hist=np.zeros((len(all_x), 0)),
        y_hist=np.zeros((len(all_x), 0)),
        t_hist=np.zeros((len(all_x), 0))
    )
    
    print(f"Dataset creado con {len(image_paths)} imágenes y {len(all_x)} fijaciones totales")
    return stimuli, fixations



def create_train_val_datasets(root_dir, transform=None, val_split=0.2):
    """Crea los datasets de entrenamiento y validación con soporte para categorías"""
    # Cargar datos
    image_paths, fixation_paths, categories, category_to_idx = load_cat2000_data(root_dir)
    
    # Crear datasets de pysaliency
    stimuli, fixations = create_pysaliency_datasets(image_paths, fixation_paths)
    
    # Crear y entrenar center bias model
    centerbias_model, baseline_ll = create_center_bias_model(stimuli, fixations)
    
    # Crear dataset completo
    full_dataset = CAT2000Dataset(
        image_paths=image_paths,
        fixation_paths=fixation_paths,
        categories=categories,
        category_to_idx=category_to_idx,
        centerbias_model=centerbias_model,
        transform=transform
    )
    
    # Realizar split estratificado para mantener distribución de categorías
    train_indices = []
    val_indices = []
    
    for category in set(categories):
        cat_indices = [i for i, cat in enumerate(categories) if cat == category]
        n_val = int(len(cat_indices) * val_split)
        
        # Shuffle los índices
        np.random.shuffle(cat_indices)
        
        val_indices.extend(cat_indices[:n_val])
        train_indices.extend(cat_indices[n_val:])
    
    # Crear splits
    train_dataset = CAT2000SplitDataset(full_dataset, train_indices)
    val_dataset = CAT2000SplitDataset(full_dataset, val_indices)
    
    print("\nDistribución del split:")
    print(f"Training: {len(train_dataset)} imágenes")
    print(f"Validation: {len(val_dataset)} imágenes")
    
    # Verificar distribución de categorías en splits
    verify_category_distribution(train_dataset, val_dataset)
    
    return train_dataset, val_dataset, baseline_ll, baseline_ll

def verify_dataset(dataset, name="Dataset"):
    """
    Verifica la integridad del dataset
    """
    print(f"\n=== Verificando {name} ===")
    print(f"Número total de imágenes: {len(dataset)}")
    
    # Verificar si estamos usando un dataset dividido
    if isinstance(dataset, CAT2000SplitDataset):
        image_paths = dataset.image_paths
        fixation_paths = dataset.fixation_paths
    else:
        image_paths = dataset.dataset.image_paths
        fixation_paths = dataset.dataset.fixation_paths
    
    # Verificar fijaciones
    n_fixations = []
    empty_images = 0
    
    for i in range(len(dataset)):
        mat_data = scipy.io.loadmat(fixation_paths[i])
        n_fix = np.sum(mat_data['fixLocs'] > 0)
        n_fixations.append(n_fix)
        
        if n_fix == 0:
            empty_images += 1
            print(f"Advertencia: Imagen {i} ({os.path.basename(image_paths[i])}) no tiene fijaciones")
    
    print("\nEstadísticas de fijaciones:")
    print(f"Media de fijaciones por imagen: {np.mean(n_fixations):.2f}")
    print(f"Mediana de fijaciones por imagen: {np.median(n_fixations):.2f}")
    print(f"Min fijaciones: {np.min(n_fixations)}")
    print(f"Max fijaciones: {np.max(n_fixations)}")
    print(f"Imágenes sin fijaciones: {empty_images}")

def verify_category_distribution(train_dataset, val_dataset):
    """Verifica la distribución de categorías en los splits"""
    print("\nDistribución de categorías:")
    print("{:<15} {:<10} {:<10}".format("Categoría", "Train", "Val"))
    print("-" * 35)
    
    cat_to_name = {v: k for k, v in train_dataset.category_to_idx.items()}
    
    train_dist = Counter(train_dataset.categories)
    val_dist = Counter(val_dataset.categories)
    
    for cat_idx in sorted(train_dataset.category_to_idx.values()):
        cat_name = cat_to_name[cat_idx]
        print("{:<15} {:<10d} {:<10d}".format(
            cat_name, 
            train_dist[cat_idx],
            val_dist[cat_idx]
        ))