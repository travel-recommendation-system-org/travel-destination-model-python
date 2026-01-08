# main.py - VERSION CORRIGÉE
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import json
import shutil
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set up paths for LOCAL environment
BASE_PATH = r"./data"
IMAGES_PATH = os.path.join(BASE_PATH, "attractions_images")
MODELS_PATH = r"models"
FEATURES_CACHE = os.path.join(MODELS_PATH, "cnn_features.pkl")
FEATURES_MERGED_CACHE = os.path.join(MODELS_PATH, "cnn_features_merged.pkl")
CHECKPOINT_DIR = os.path.join(MODELS_PATH, "cnn_checkpoints")
# Create directories
os.makedirs(MODELS_PATH, exist_ok=True)

print(f"Data path: {BASE_PATH}")
print(f"Models path: {MODELS_PATH}")

# Import our modules
from data_loader import TourismDataLoader
from ncf_model import NCFModel
from cnn_features import create_incremental_cnn_extractor
from hybrid_recommender import HybridRecommender
from evaluator import Evaluator

# Fonction pour vérifier si les features CNN existent déjà
def check_existing_cnn_features():
    """Vérifier si les features CNN existent déjà"""
    existing_files = []

    # Vérifier les fichiers de features
    if os.path.exists(FEATURES_CACHE):
        existing_files.append(("Main features", FEATURES_CACHE))

    if os.path.exists(FEATURES_MERGED_CACHE):
        existing_files.append(("Merged features", FEATURES_MERGED_CACHE))

    # Vérifier les checkpoints
    if os.path.exists(CHECKPOINT_DIR):
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR)
                          if f.endswith('.pkl') or f.endswith('.json')]
        if checkpoint_files:
            existing_files.append(("Checkpoints", f"{CHECKPOINT_DIR} ({len(checkpoint_files)} fichiers)"))

    return existing_files

# Fonction pour charger les features existantes
def load_existing_features():
    """Charger les features CNN existantes"""
    # Essayer d'abord le fichier merged
    if os.path.exists(FEATURES_MERGED_CACHE):
        try:
            print(f"Loading features from merged cache: {FEATURES_MERGED_CACHE}")
            with open(FEATURES_MERGED_CACHE, 'rb') as f:
                features = pickle.load(f)

            # Valider les features
            if features and len(features) > 0:
                sample_key = next(iter(features.keys()))
                sample_feature = features[sample_key]

                if not np.isnan(sample_feature).any() and np.linalg.norm(sample_feature) > 0.1:
                    print(f"Successfully loaded {len(features)} features from merged cache")
                    return features
        except Exception as e:
            print(f"Error loading merged cache: {e}")

    # Essayer le fichier principal
    if os.path.exists(FEATURES_CACHE):
        try:
            print(f"Loading features from main cache: {FEATURES_CACHE}")
            with open(FEATURES_CACHE, 'rb') as f:
                features = pickle.load(f)

            # Valider les features
            if features and len(features) > 0:
                sample_key = next(iter(features.keys()))
                sample_feature = features[sample_key]

                if not np.isnan(sample_feature).any() and np.linalg.norm(sample_feature) > 0.1:
                    print(f"Successfully loaded {len(features)} features from main cache")
                    return features
        except Exception as e:
            print(f"Error loading main cache: {e}")

    return None

def main(use_existing_features=True):
    """
    Main execution function - VERSION CORRIGÉE
    
    Args:
        use_existing_features: Si True, utilise les features existantes si disponibles
    """
    print("=" * 70)
    print("TOURISM HYBRID RECOMMENDATION SYSTEM - LOCAL VERSION")
    print("Marrakech Attractions - NCF + CNN")
    print("=" * 70)

    # Initialiser les variables
    attractions_features = None
    cnn_extractor = None
    skip_cnn_extraction = False

    # Vérifier si les features CNN existent déjà
    print("\n" + "=" * 60)
    print("CHECKING FOR EXISTING CNN FEATURES")
    print("=" * 60)

    if use_existing_features:
        existing_features = check_existing_cnn_features()

        if existing_features:
            print("Found existing CNN features:")
            for name, path in existing_features:
                if os.path.isfile(path):
                    file_size = os.path.getsize(path) / (1024*1024)
                    print(f"  ✓ {name}: {path} ({file_size:.1f} MB)")
                else:
                    print(f"  ✓ {name}: {path}")

            # Essayer de charger les features existantes
            attractions_features = load_existing_features()
            if attractions_features is not None:
                print(f"\n✓ Using existing CNN features: {len(attractions_features)} attractions")
                skip_cnn_extraction = True
            else:
                print("\n✗ Existing features are invalid or empty. Will extract new features.")
                skip_cnn_extraction = False
        else:
            print("No existing CNN features found.")
            skip_cnn_extraction = False
    else:
        print("Skipping existing features check (force new extraction)")
        skip_cnn_extraction = False

    # Vérifier et optimiser GPU
    print("\n" + "=" * 60)
    print("CHECKING GPU")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("GPU Optimization Settings:")
        print(f"  CUDA Available: Yes")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()
        print("  cuDNN benchmark enabled")
        gpu_available = True
    else:
        print("WARNING: No GPU available. Using CPU (slow)")
        gpu_available = False

    # Step 1: Load and prepare data
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    data_loader = TourismDataLoader(BASE_PATH)
    reviews_df, attractions_df = data_loader.load_data()

    # Display data info
    print("\nData Overview:")
    print(f"  Total users: {data_loader.num_users}")
    print(f"  Total attractions: {data_loader.num_attractions}")
    print(f"  Total interactions: {len(data_loader.interactions)}")

    # Analyser la distribution des données
    data_loader.analyze_data_distribution()

    # Check image availability
    print("\nChecking image availability...")
    image_check = data_loader.check_image_availability(n_samples=10)

    # Split data
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split(
        test_size=0.2, random_state=42
    )

    # Step 2: Train NCF Model
    print("\n" + "=" * 60)
    print("STEP 2: Training NCF Model")
    print("=" * 60)

    ncf_model = NCFModel(
        num_users=data_loader.num_users,
        num_attractions=data_loader.num_attractions,
        embedding_dim=50,
        mlp_layers=[64, 32, 16]
    )

    ncf_model.build_model()

    # Train the model
    print("\nTraining NCF model...")
    history = ncf_model.train(
        X_train, y_train,
        epochs=15,
        batch_size=256,
        validation_split=0.1
    )

    # Save NCF model
    ncf_save_path = os.path.join(MODELS_PATH, "ncf_model")
    ncf_model.save_model(ncf_save_path)
    print(f"✓ NCF model saved to {ncf_save_path}")

    # Step 3: Extract CNN Features (seulement si nécessaire)
    print("\n" + "=" * 60)
    print("STEP 3: CNN Feature Extraction")
    print("=" * 60)

    if skip_cnn_extraction and attractions_features is not None:
        print("\n✓ SKIPPING CNN extraction - using existing features")
        print(f"  Features loaded: {len(attractions_features)} attractions")

        # Créer un extracteur minimal pour la compatibilité
        class SimpleCNNExtractor:
            def __init__(self, features):
                self.attractions_features = features

            def find_similar_attractions(self, target_features, features_dict=None, k=10):
                if features_dict is None:
                    features_dict = self.attractions_features

                from sklearn.metrics.pairwise import cosine_similarity
                attraction_ids = list(features_dict.keys())
                features_matrix = np.array([features_dict[aid] for aid in attraction_ids])

                target_features = target_features.reshape(1, -1)
                similarities = cosine_similarity(target_features, features_matrix)[0]

                top_indices = np.argsort(similarities)[::-1][:k]
                return [(attraction_ids[i], similarities[i]) for i in top_indices]

            def find_similar_by_image(self, image_path, features_dict=None, k=10):
                # Pour simplifier, retourner vide
                return []

        cnn_extractor = SimpleCNNExtractor(attractions_features)
    else:
        print("\nStarting CNN feature extraction...")

        # Supprimer les anciens fichiers si on force une nouvelle extraction
        if not skip_cnn_extraction and not use_existing_features:
            print("Cleaning up old feature files...")
            for file_path in [FEATURES_CACHE, FEATURES_MERGED_CACHE]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"  Removed: {file_path}")
            
            if os.path.exists(CHECKPOINT_DIR):
                shutil.rmtree(CHECKPOINT_DIR)
                print(f"  Removed: {CHECKPOINT_DIR}")

        # Créer l'extracteur avec sauvegarde incrémentielle
        cnn_extractor = create_incremental_cnn_extractor()

        print(f"\nCheckpoint directory: {CHECKPOINT_DIR}")
        print(f"Features cache: {FEATURES_CACHE}")

        # Extraire avec sauvegarde incrémentielle
        print("\nStarting incremental feature extraction...")
        print("Features will be saved every 100 attractions.")
        print("If interrupted, run again to resume from last checkpoint.")

        attractions_features = cnn_extractor.extract_all_attractions_features_incremental(
            data_loader=data_loader,
            cache_path=FEATURES_CACHE,
            checkpoint_dir=CHECKPOINT_DIR,
            batch_size=100
        )

        # Vérifier la sauvegarde
        if os.path.exists(FEATURES_CACHE):
            file_size = os.path.getsize(FEATURES_CACHE) / (1024*1024)
            print(f"\n✓ Features saved successfully: {FEATURES_CACHE}")
            print(f"  File size: {file_size:.1f} MB")
            print(f"  Number of features: {len(attractions_features)}")

    # Step 4: Create Recommender System
    print("\n" + "=" * 60)
    print("STEP 4: CREATING RECOMMENDER SYSTEM")
    print("=" * 60)

    # Créer le recommandateur hybride
    print("\nCreating hybrid recommender (NCF + CNN)...")

    hybrid_recommender = HybridRecommender(
        ncf_model=ncf_model,
        cnn_extractor=cnn_extractor,
        data_loader=data_loader,
        attractions_features=attractions_features,
        alpha=0.7
    )

    print(f"✓ Hybrid recommender created with alpha = {hybrid_recommender.alpha}")

    # Step 5: Evaluation
    print("\n" + "=" * 60)
    print("STEP 5: COMPREHENSIVE EVALUATION")
    print("=" * 60)

    evaluator = Evaluator()

    # Generate evaluation report
    print("\nGenerating evaluation report...")
    try:
        report = evaluator.generate_evaluation_report(
            ncf_model=ncf_model,
            hybrid_recommender=hybrid_recommender,
            data_loader=data_loader,
            X_test=X_test,
            y_test=y_test
        )
        print("✓ Evaluation report generated")
    except Exception as e:
        print(f"✗ Error generating evaluation report: {e}")
        report = {}

    # Step 7: System Insights and Summary
    print("\n" + "=" * 70)
    print("STEP 7: SYSTEM INSIGHTS AND SUMMARY")
    print("=" * 70)

    # Final summary
    print("\n" + "=" * 40)
    print("SYSTEM SUMMARY")
    print("=" * 40)
    print(f"  • Data loaded: {data_loader.num_users} users, {data_loader.num_attractions} attractions")
    print(f"  • NCF model trained: RMSE = {report.get('ncf_metrics', {}).get('rmse_rating', 'N/A'):.2f} (1-5 scale)")
    print(f"  • CNN features: {len(attractions_features) if attractions_features else 0} attractions")
    print(f"  • Hybrid recommender: READY (2 components)")
    print(f"     - Alpha (NCF vs CNN): {hybrid_recommender.alpha}")
    print("=" * 40)

    # Return system components
    system_components = {
        'data_loader': data_loader,
        'ncf_model': ncf_model,
        'cnn_extractor': cnn_extractor,
        'hybrid_recommender': hybrid_recommender,
        'evaluator': evaluator,
        'attractions_features': attractions_features
    }

    return system_components

# Execute the main function
if __name__ == "__main__":
    print("\nStarting system execution...")
    try:
        system_components = main(use_existing_features=True)

        print("\n" + "=" * 60)
        print("SYSTEM READY FOR USE")
        print("=" * 60)

        print("\nSystem components available in 'system_components' variable:")
        print("  • data_loader: Data loading and preprocessing")
        print("  • ncf_model: Neural Collaborative Filtering model")
        print("  • cnn_extractor: CNN feature extractor for images")
        print("  • hybrid_recommender: Hybrid recommender (NCF + CNN)")
        print("  • evaluator: Evaluation metrics and analysis")
        print("  • attractions_features: CNN features dictionary")

        print("\n" + "=" * 50)
        print("QUICK USAGE EXAMPLES")
        print("=" * 50)

        print("""
# 1. Get recommendations for a user
recs = system_components['hybrid_recommender'].recommend_for_user(user_id=100, k=5)

# 2. Visual search by image
visual_recs = system_components['hybrid_recommender'].recommend_by_image(
    image_path="path/to/image.jpg",
    k=5
)

# 3. Change recommendation weights (alpha = NCF vs CNN weight)
system_components['hybrid_recommender'].set_alpha(0.5)  # 50% NCF, 50% CNN

# 4. Analyze recommendations
analysis = system_components['hybrid_recommender'].analyze_recommendations()

# 5. Get user history
user_history = system_components['data_loader'].get_user_history(user_id=100)
        """)

    except Exception as e:
        print(f"\nError during system execution: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. All data files are in the correct location")
        print("2. Required modules are installed")
        print("3. Paths are correctly configured")