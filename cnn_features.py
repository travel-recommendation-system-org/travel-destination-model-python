# cnn_features.py (avec sauvegarde incrémentielle) - VERSION CORRIGÉE
import warnings
warnings.filterwarnings('ignore')

# IMPORTANT: Installer la bonne version de sympy d'abord
try:
    import sympy
    sympy_version = sympy.__version__
    print(f"SymPy version: {sympy_version}")
except:
    print("Installing correct sympy version...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sympy==1.12"])

# Maintenant importer les autres modules
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pickle
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import gc
import shutil

# Gestion de autocast
try:
    from torch.cuda.amp import autocast
    AUTOCAST_AVAILABLE = True
except ImportError:
    AUTOCAST_AVAILABLE = False
    print("Autocast not available, using standard precision")

class IncrementalCNNFeatureExtractor:
    """
    CNN Feature Extractor avec sauvegarde incrémentielle et reprise
    """

    def __init__(self, model_name='resnet50', device='cuda', batch_size=512, num_workers=15):
        """
        Initialisation avec sauvegarde incrémentielle
        """
        self.model_name = model_name
        # Vérifier si CUDA est disponible
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            device = 'cpu'

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Optimisations PyTorch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        print(f"Initializing Incremental CNN Feature Extractor")
        print(f"Device: {self.device}, Model: {model_name}")
        print(f"Batch size: {batch_size}, Workers: {num_workers}")

        # Charger le modèle
        self.model = self._load_model()
        self.model.eval()
        self.model = self.model.to(self.device)

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Caches
        self.feature_cache = {}
        self.image_cache = {}
        self.cache_lock = Lock()

        # État de progression
        self.progress_file = None
        self.checkpoint_dir = None

        # Info GPU
        if 'cuda' in str(self.device):
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
            except:
                print("GPU information not available")

    def _load_model(self):
        """Charger le modèle avec compatibilité des versions"""
        try:
            # Essayer avec la nouvelle API des poids
            if self.model_name == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            elif self.model_name == 'resnet18':
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            # Fallback à l'ancienne API
            print("Using legacy API for model loading")
            if self.model_name == 'resnet50':
                model = models.resnet50(pretrained=True)
            elif self.model_name == 'resnet18':
                model = models.resnet18(pretrained=True)
            else:
                model = models.resnet50(pretrained=True)

        # Retirer la dernière couche (classificateur)
        model = nn.Sequential(*list(model.children())[:-1])

        for param in model.parameters():
            param.requires_grad = False

        return model

    def _load_images_batch(self, image_paths):
        """Charger un batch d'images"""
        results = []
        valid_paths = []

        def load_single_image(img_path):
            try:
                with self.cache_lock:
                    if img_path in self.image_cache:
                        return img_path, self.image_cache[img_path]

                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    tensor = self.transform(img)

                with self.cache_lock:
                    self.image_cache[img_path] = tensor

                return img_path, tensor
            except Exception as e:
                # print(f"Error loading image {img_path}: {e}")
                return img_path, None

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_single_image, path): path for path in image_paths}

            for future in as_completed(futures):
                img_path, tensor = future.result()
                if tensor is not None:
                    valid_paths.append(img_path)
                    results.append(tensor)

        if results:
            return torch.stack(results), valid_paths
        return None, []

    def extract_features_batch(self, image_paths):
        """Extraire les features d'un batch"""
        if not image_paths:
            return []

        all_features = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]

            batch_tensors, valid_paths = self._load_images_batch(batch_paths)
            if batch_tensors is None:
                continue

            batch_tensors = batch_tensors.to(self.device, non_blocking=True)

            # Utiliser autocast si disponible
            if AUTOCAST_AVAILABLE and 'cuda' in str(self.device):
                with torch.no_grad(), autocast():
                    features = self.model(batch_tensors)
                    features = features.squeeze()

                    if len(features.shape) == 1:
                        features = features.unsqueeze(0)

                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                    features = features.cpu().numpy()
            else:
                with torch.no_grad():
                    features = self.model(batch_tensors)
                    features = features.squeeze()

                    if len(features.shape) == 1:
                        features = features.unsqueeze(0)

                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                    features = features.cpu().numpy()

            all_features.extend(features)

            del batch_tensors
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

        return all_features

    def extract_attraction_features(self, attraction_id, data_loader):
        """Extraire les features pour une attraction"""
        if attraction_id in self.feature_cache:
            return self.feature_cache[attraction_id]

        image_paths = data_loader.get_attraction_image_paths(attraction_id)

        if not image_paths:
            dim = 2048 if 'resnet50' in self.model_name else 512
            random_features = np.random.randn(dim)
            result = random_features / np.linalg.norm(random_features)
            self.feature_cache[attraction_id] = result
            return result

        image_paths = image_paths[:3]

        features_list = self.extract_features_batch(image_paths)

        if features_list:
            avg_features = np.mean(features_list, axis=0)
            norm = np.linalg.norm(avg_features)
            if norm > 0:
                avg_features = avg_features / norm
        else:
            dim = 2048 if 'resnet50' in self.model_name else 512
            random_features = np.random.randn(dim)
            avg_features = random_features / np.linalg.norm(random_features)

        self.feature_cache[attraction_id] = avg_features

        return avg_features

    def setup_checkpoint(self, checkpoint_dir):
        """Configurer le système de checkpoint"""
        self.checkpoint_dir = checkpoint_dir
        self.progress_file = os.path.join(checkpoint_dir, "extraction_progress.json")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Progress file: {self.progress_file}")

    def save_progress(self, completed_ids, features_dict, batch_num, total_batches):
        """Sauvegarder la progression - VERSION CORRIGÉE"""
        if not self.checkpoint_dir:
            return

        # Sauvegarder les features de ce batch
        batch_file = os.path.join(self.checkpoint_dir, f"features_batch_{batch_num:04d}.pkl")
        with open(batch_file, 'wb') as f:
            pickle.dump({att_id: features_dict[att_id] for att_id in completed_ids}, f)

        # Convertir numpy int64 en int Python
        completed_ids_list = [int(att_id) for att_id in completed_ids]

        # Sauvegarder l'état de progression
        progress_state = {
            'batch_num': int(batch_num),
            'total_batches': int(total_batches),
            'completed_ids': completed_ids_list,
            'timestamp': float(time.time()),
            'total_completed': int(len(completed_ids))
        }

        with open(self.progress_file, 'w') as f:
            json.dump(progress_state, f, indent=2)

        print(f"Progress saved: Batch {batch_num}/{total_batches}, {len(completed_ids)} attractions")

    def load_progress(self):
        """Charger la progression précédente"""
        if not self.progress_file or not os.path.exists(self.progress_file):
            return None, {}, 0

        try:
            with open(self.progress_file, 'r') as f:
                progress_state = json.load(f)

            print(f"Found previous progress: Batch {progress_state['batch_num']}/{progress_state['total_batches']}")
            print(f"Previously completed: {progress_state['total_completed']} attractions")

            # Charger tous les batches précédents
            features_dict = {}
            for batch_num in range(progress_state['batch_num'] + 1):
                batch_file = os.path.join(self.checkpoint_dir, f"features_batch_{batch_num:04d}.pkl")
                if os.path.exists(batch_file):
                    with open(batch_file, 'rb') as f:
                        batch_features = pickle.load(f)
                        features_dict.update(batch_features)

            # Mettre à jour le cache
            self.feature_cache.update(features_dict)

            return progress_state, features_dict, progress_state['batch_num']

        except Exception as e:
            print(f"Error loading progress: {e}")
            return None, {}, 0

    def merge_checkpoint_files(self, output_path):
        """Fusionner tous les fichiers checkpoint en un seul"""
        if not self.checkpoint_dir:
            return None

        print(f"Merging checkpoint files from {self.checkpoint_dir}")

        all_features = {}
        batch_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('features_batch_') and f.endswith('.pkl')]
        batch_files.sort()

        for batch_file in tqdm(batch_files, desc="Merging batches"):
            batch_path = os.path.join(self.checkpoint_dir, batch_file)
            try:
                with open(batch_path, 'rb') as f:
                    batch_features = pickle.load(f)
                    all_features.update(batch_features)
            except Exception as e:
                print(f"Error loading {batch_file}: {e}")

        # Sauvegarder le fichier fusionné
        with open(output_path, 'wb') as f:
            pickle.dump(all_features, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Merged {len(all_features)} features into {output_path}")

        return all_features

    def extract_all_attractions_features_incremental(self, data_loader, cache_path=None, checkpoint_dir=None, batch_size=100):
        """
        Extraction avec sauvegarde incrémentielle
        """
        # Configurer les checkpoints
        if checkpoint_dir:
            self.setup_checkpoint(checkpoint_dir)

        # Vérifier si on peut reprendre
        progress_state, loaded_features, start_batch = self.load_progress()

        # Obtenir toutes les attractions
        attractions_ids = data_loader.attractions_df['attraction_id'].unique()
        total_attractions = len(attractions_ids)

        # Si on a chargé une progression, filtrer les IDs déjà traités
        if loaded_features:
            remaining_ids = [att_id for att_id in attractions_ids if att_id not in loaded_features]
            print(f"Resuming from checkpoint: {len(loaded_features)} already processed, {len(remaining_ids)} remaining")
            attractions_ids = remaining_ids
            features_dict = loaded_features
        else:
            features_dict = {}

        # Diviser en batches pour le checkpointing
        batches = []
        for i in range(0, len(attractions_ids), batch_size):
            batches.append(attractions_ids[i:i + batch_size])

        total_batches = len(batches)

        print(f"\nStarting incremental feature extraction")
        print(f"Total attractions: {total_attractions}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {total_batches}")
        print(f"Starting from batch: {start_batch}")

        start_time = time.time()
        completed_count = len(features_dict)

        # Traiter chaque batch
        for batch_num in range(start_batch, total_batches):
            batch_ids = batches[batch_num]
            batch_start_time = time.time()

            print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch_ids)} attractions)")

            # Traiter les attractions de ce batch
            batch_features = {}

            with tqdm(total=len(batch_ids), desc=f"Batch {batch_num + 1}") as pbar:
                for att_id in batch_ids:
                    try:
                        features = self.extract_attraction_features(att_id, data_loader)
                        batch_features[att_id] = features
                        completed_count += 1

                        # Afficher la progression globale
                        elapsed = time.time() - start_time
                        speed = completed_count / elapsed if elapsed > 0 else 0
                        pbar.set_postfix({
                            "total": completed_count,
                            "speed": f"{speed:.1f} att/s"
                        })

                    except Exception as e:
                        print(f"\nError processing attraction {att_id}: {e}")
                        # Fallback
                        dim = 2048 if 'resnet50' in self.model_name else 512
                        random_features = np.random.randn(dim)
                        batch_features[att_id] = random_features / np.linalg.norm(random_features)
                        completed_count += 1

                    pbar.update(1)

            # Ajouter les features du batch au dictionnaire global
            features_dict.update(batch_features)

            # Sauvegarder le checkpoint
            if self.checkpoint_dir:
                self.save_progress(batch_ids, features_dict, batch_num, total_batches)

            batch_time = time.time() - batch_start_time
            print(f"Batch {batch_num + 1} completed in {batch_time:.1f}s")

            # Nettoyer périodiquement
            if batch_num % 5 == 0 and 'cuda' in str(self.device):
                torch.cuda.empty_cache()
                gc.collect()

        elapsed_time = time.time() - start_time

        print(f"\nExtraction completed!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Total attractions processed: {len(features_dict)}")
        print(f"Average speed: {len(features_dict)/elapsed_time:.1f} attractions/second")

        # Sauvegarder le cache final
        if cache_path:
            try:
                print(f"\nSaving final features to {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(features_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Features saved successfully")

                # Si on a des checkpoints, les fusionner
                if self.checkpoint_dir:
                    merged_path = cache_path.replace('.pkl', '_merged.pkl')
                    self.merge_checkpoint_files(merged_path)

            except Exception as e:
                print(f"Error saving cache: {e}")

        return features_dict

    # Méthodes de compatibilité
    def extract_all_attractions_features(self, data_loader, attractions_ids=None, cache_path=None, use_parallel=True):
        """Interface standard avec checkpointing"""
        checkpoint_dir = None
        if cache_path:
            checkpoint_dir = os.path.join(os.path.dirname(cache_path), "checkpoints")

        return self.extract_all_attractions_features_incremental(
            data_loader=data_loader,
            cache_path=cache_path,
            checkpoint_dir=checkpoint_dir,
            batch_size=100
        )

    def extract_all_attractions_features_gpu(self, data_loader, attractions_ids=None, cache_path=None):
        """Interface GPU avec checkpointing"""
        return self.extract_all_attractions_features(data_loader, attractions_ids, cache_path)

    # Méthodes utilitaires
    def compute_similarity(self, features1, features2):
        from sklearn.metrics.pairwise import cosine_similarity
        f1 = features1.reshape(1, -1)
        f2 = features2.reshape(1, -1)
        return cosine_similarity(f1, f2)[0][0]

    def find_similar_attractions(self, target_features, features_dict, k=10):
        if not features_dict:
            return []

        attraction_ids = list(features_dict.keys())
        features_matrix = np.array([features_dict[aid] for aid in attraction_ids])

        target_features = target_features.reshape(1, -1)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(target_features, features_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:k]
        return [(attraction_ids[i], similarities[i]) for i in top_indices]

    def find_similar_by_image(self, input_image_path, features_dict, k=10):
        features_list = self.extract_features_batch([input_image_path])
        if features_list:
            target_features = features_list[0]
            return self.find_similar_attractions(target_features, features_dict, k)
        return []

    def validate_features(self, features_dict, n_samples=5):
        print("\nFeature Validation:")
        print("-" * 40)

        if not features_dict:
            print("No features to validate.")
            return

        sample_ids = list(features_dict.keys())[:n_samples]

        for i, att_id in enumerate(sample_ids, 1):
            features = features_dict[att_id]
            print(f"\nSample {i} (Attraction ID: {att_id}):")
            print(f"  Shape: {features.shape}")
            print(f"  Mean: {features.mean():.6f}")
            print(f"  Std: {features.std():.6f}")

        print(f"\nSummary:")
        print(f"  Total attractions: {len(features_dict)}")

        return {'total': len(features_dict)}

    def clear_cache(self):
        with self.cache_lock:
            self.feature_cache.clear()
            self.image_cache.clear()

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        print("All caches cleared")

def create_incremental_cnn_extractor():
    """Créer un extracteur avec sauvegarde incrémentielle"""
    print("\n" + "="*60)
    print("Creating Incremental CNN Feature Extractor")
    print("="*60)

    # Vérifier et installer sympy si nécessaire
    try:
        import sympy
        sympy_version = sympy.__version__
        print(f"SymPy version: {sympy_version}")
        if sympy_version != '1.12':
            print("Installing sympy 1.12 for compatibility...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sympy==1.12", "--quiet"])
    except:
        print("Installing sympy 1.12...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sympy==1.12", "--quiet"])

    if not torch.cuda.is_available():
        print("WARNING: No GPU available. Using CPU mode.")
        return IncrementalCNNFeatureExtractor(
            model_name='resnet18',
            device='cpu',
            batch_size=32,
            num_workers=8
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"GPU: {gpu_name}")
    print(f"GPU memory: {gpu_memory:.1f} GB")

    # Ajuster les paramètres selon le GPU
    if 'T4' in gpu_name or 'P100' in gpu_name:
        print("Config: batch_size = 256, num_workers = 6")
        batch_size = 256
        num_workers = 6
    elif 'V100' in gpu_name or 'A100' in gpu_name:
        print("Config: batch_size = 512, num_workers = 8")
        batch_size = 512
        num_workers = 8
    else:
        print("Config: batch_size = 128, num_workers = 4")
        batch_size = 128
        num_workers = 4

    return IncrementalCNNFeatureExtractor(
        model_name='resnet50',
        device='cuda',
        batch_size=batch_size,
        num_workers=num_workers
    )