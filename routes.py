# routes.py - VERSION COMPLÈTE AVEC TOUTES LES ROUTES
from flask import Flask, request, jsonify, send_file
import numpy as np
import pandas as pd
import os
from PIL import Image
import io
import base64
import json
import pickle
import tempfile
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
import torch

# Import des composants du système
from data_loader import TourismDataLoader
from cold_start_solver import create_cold_start_solver

# Configuration
app = Flask(__name__)
BASE_PATH = r"./data"
MODELS_PATH = r"models"

# Variables globales pour les composants du système
system_components = None
cold_start_solver = None
system_initialized = False

# Configuration pour le téléchargement d'images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    """Vérifier si le fichier a une extension autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_custom_cnn_extractor():
    """
    Créer un extracteur CNN compatible avec les features existantes
    """
    print("\n" + "="*50)
    print("Creating Custom CNN Extractor")
    print("="*50)
    
    class CustomCNNExtractor:
        def __init__(self):
            self.model_name = "resnet50_custom"
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.batch_size = 32
            print(f"✓ Created CustomCNNExtractor on {self.device}")
            
            # Charger le modèle
            self._load_model()
        
        def _load_model(self):
            """Charger le modèle ResNet50"""
            try:
                import torchvision.models as models
                import torch.nn as nn
                import torchvision.transforms as transforms
                
                self.model = models.resnet50(pretrained=True)
                self.model = nn.Sequential(*list(self.model.children())[:-1])
                self.model.eval()
                self.model = self.model.to(self.device)
                
                # Transformation pour les images
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                print(f"✓ Model loaded and moved to {self.device}")
                
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                self.model = None
                self.transform = None
        
        def extract_features_batch(self, image_paths):
            """Extraire les features d'un batch d'images"""
            if not self.model:
                raise ValueError("Model not loaded")
            
            features_list = []
            
            for img_path in image_paths:
                try:
                    # Charger et transformer l'image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Extraire les features
                    with torch.no_grad():
                        features = self.model(img_tensor)
                        features = features.squeeze()
                        features = torch.nn.functional.normalize(features, p=2, dim=0)
                        features_list.append(features.cpu().numpy())
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    # Fallback: features aléatoires
                    features_list.append(np.random.randn(2048))
            
            return features_list
        
        def find_similar_attractions(self, target_features, features_dict, k=10):
            """Trouver des attractions similaires"""
            from sklearn.metrics.pairwise import cosine_similarity
            
            if not features_dict:
                return []
            
            # Convertir en matrices
            att_ids = list(features_dict.keys())
            features_matrix = np.array([features_dict[aid] for aid in att_ids])
            
            # Calculer les similarités
            similarities = cosine_similarity([target_features], features_matrix)[0]
            
            # Obtenir les k meilleures
            top_indices = np.argsort(similarities)[::-1][:k]
            
            return [(att_ids[i], similarities[i]) for i in top_indices]
    
    return CustomCNNExtractor()

def init_system():
    """
    Initialiser le système de recommandation
    """
    global system_components, cold_start_solver, system_initialized
    
    print("=" * 60)
    print("Initializing recommendation system...")
    print("=" * 60)
    
    try:
        # 1. CHARGER LES FEATURES CNN EXISTANTES
        features_path = os.path.join(MODELS_PATH, "cnn_features.pkl")
        
        if os.path.exists(features_path):
            print(f"Loading CNN features from: {features_path}")
            with open(features_path, 'rb') as f:
                attractions_features = pickle.load(f)
            print(f"✓ Loaded {len(attractions_features)} CNN features")
        else:
            print(f"✗ CNN features not found at: {features_path}")
            return False
        
        # 2. CHARGER LES AUTRES COMPOSANTS
        print("\nLoading other components...")
        from main import main
        
        # Utiliser le paramètre pour éviter de recalculer les features
        system_components = main(use_existing_features=True)
        
        if system_components is None:
            print("✗ Failed to load system components")
            return False
        
        # 3. REMPLACER L'EXTRACTEUR CNN
        print("\n" + "="*40)
        print("Setting up CNN extractor")
        print("="*40)
        
        # Créer notre extracteur personnalisé
        custom_extractor = create_custom_cnn_extractor()
        
        # Remplacer l'extracteur
        system_components['cnn_extractor'] = custom_extractor
        
        # Mettre à jour les features
        system_components['attractions_features'] = attractions_features
        
        # 4. INITIALISER LE COLD START SOLVER
        cold_start_solver = create_cold_start_solver(
            system_components['data_loader'],
            attractions_features
        )
        
        system_initialized = True
        
        print("\n" + "="*60)
        print("SYSTEM INITIALIZATION COMPLETE")
        print("="*60)
        print(f"✓ Users: {system_components['data_loader'].num_users}")
        print(f"✓ Attractions: {system_components['data_loader'].num_attractions}")
        print(f"✓ CNN Features: {len(attractions_features)}")
        print(f"✓ CNN Extractor: {type(custom_extractor).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error initializing system: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.before_request
def before_request_handler():
    """Vérifier si le système est initialisé"""
    global system_initialized
    
    if request.endpoint == 'health_check':
        return
    
    if not system_initialized:
        print("System not initialized, initializing now...")
        init_system()

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_attraction_image_paths(attraction_id):
    """
    Obtenir les chemins d'images pour une attraction
    """
    if not system_initialized or system_components is None:
        return []
    
    try:
        data_loader = system_components['data_loader']
        image_paths = data_loader.get_attraction_image_paths(attraction_id)
        
        relative_paths = []
        for path in image_paths:
            if os.path.exists(path):
                normalized_path = os.path.normpath(path)
                relative_paths.append(normalized_path)
        
        return relative_paths[:5]
        
    except Exception as e:
        print(f"Error getting image paths for attraction {attraction_id}: {e}")
        return []

def get_attraction_details(attraction_id):
    """
    Obtenir les détails d'une attraction avec chemins d'images
    VERSION CORRIGÉE: toutes les valeurs converties en types JSON-serializable
    """
    try:
        if system_components is None or 'data_loader' not in system_components:
            return {
                'name': f'Attraction {attraction_id}',
                'category': 'Unknown',
                'rating': 0.0,
                'image_paths': [],
                'image_count': 0,
                'first_image': None,
                'has_images': False
            }
        
        info = system_components['data_loader'].get_attraction_info(attraction_id)
        
        if info is None or info.empty:
            return {
                'name': f'Attraction {attraction_id}',
                'category': 'Unknown',
                'rating': 0.0,
                'image_paths': [],
                'image_count': 0,
                'first_image': None,
                'has_images': False
            }
        
        # Nettoyer le nom
        name = str(info.get('attraction_url', f'Attraction {attraction_id}'))
        if isinstance(name, str):
            if 'Reviews-' in name:
                name = name.split('Reviews-')[1]
            if '.html' in name:
                name = name.replace('.html', '')
            name = name.replace('_', ' ').replace('-', ' ').title()
        else:
            name = f'Attraction {attraction_id}'
        
        # Catégorie
        category = info.get('category', 'Unknown')
        if pd.isna(category):
            category = 'Not specified'
        category = str(category)
        
        # Note
        rating = info.get('rating', 0)
        if pd.isna(rating):
            rating = 0
        rating = float(rating)
        
        # Obtenir les chemins d'images
        image_paths = get_attraction_image_paths(attraction_id)
        has_images = len(image_paths) > 0
        image_count = len(image_paths)
        first_image = image_paths[0] if image_paths else None
        
        # Créer le dictionnaire
        details = {
            'name': str(name)[:100],
            'category': str(category),
            'rating': float(rating),
            'image_count': int(image_count),
            'image_paths': list(image_paths),
            'first_image': str(first_image) if first_image else None,
            'has_images': bool(has_images)
        }
        
        # Coordonnées si disponibles
        if 'latitude' in info and 'longitude' in info:
            try:
                lat = info.get('latitude')
                lon = info.get('longitude')
                if not pd.isna(lat) and not pd.isna(lon):
                    details['latitude'] = float(lat)
                    details['longitude'] = float(lon)
            except:
                pass
        
        # Adresse si disponible
        if 'address' in info:
            address = info.get('address')
            if not pd.isna(address):
                details['address'] = str(address)
        
        return details
        
    except Exception as e:
        print(f"Error getting details for attraction {attraction_id}: {e}")
        return {
            'name': f'Attraction {attraction_id}',
            'category': 'Unknown',
            'rating': 0.0,
            'image_paths': [],
            'image_count': 0,
            'first_image': None,
            'has_images': False
        }

# ============================================================================
# ENDPOINTS PRINCIPAUX
# ============================================================================

@app.route('/')
def index():
    """Page d'accueil de l'API"""
    return jsonify({
        'message': 'Tourism Recommendation System API',
        'version': '2.0',
        'status': 'initialized' if system_initialized else 'not_initialized',
        'endpoints': {
            '/': 'API Home',
            '/recommend/user/<user_id>': 'Recommendations personnalisées',
            '/recommend/cold-start': 'Recommandations démarrage à froid',
            '/recommend/visual': 'Recherche visuelle',
            '/recommend/similar/<attraction_id>': 'Attractions similaires',
            '/recommend/popular': 'Attractions populaires',
            '/recommend/hybrid': 'Recommandations hybrides',
            '/user/history/<user_id>': 'Historique utilisateur',
            '/attraction/info/<attraction_id>': 'Informations attraction',
            '/system/status': 'Statut du système',
            '/health': 'Vérification de santé',
            '/search/by-image': 'Recherche par image (upload)'
        }
    })

@app.route('/search/by-image', methods=['POST'])
def search_by_image():
    """
    Recherche par image upload
    """
    if not system_initialized:
        return jsonify({'error': 'System not initialized', 'success': False}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided', 'success': False}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected', 'success': False}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid image format', 'success': False}), 400
        
        k = min(int(request.args.get('k', 10)), 50)
        threshold = float(request.args.get('threshold', 0.5))
        
        print(f"\n=== IMAGE SEARCH ===")
        print(f"File: {image_file.filename}, k={k}, threshold={threshold}")
        
        # Sauvegarder temporairement
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, secure_filename(image_file.filename))
        image_file.save(image_path)
        
        try:
            # Extraire les features
            cnn_extractor = system_components['cnn_extractor']
            attractions_features = system_components['attractions_features']
            
            features_list = cnn_extractor.extract_features_batch([image_path])
            
            if not features_list:
                raise ValueError("No features extracted")
            
            image_features = features_list[0]
            
            # Trouver des attractions similaires
            similar_attractions = cnn_extractor.find_similar_attractions(
                image_features,
                attractions_features,
                k=k * 2
            )
            
            # Filtrer par seuil
            filtered = [(att_id, score) for att_id, score in similar_attractions if score >= threshold]
            filtered = filtered[:k]
            
            # Formater les résultats
            results = []
            for att_id, similarity in filtered:
                details = get_attraction_details(att_id)
                
                result_item = {
                    'id': int(att_id),
                    'name': str(details.get('name', f'Attraction {att_id}')),
                    'category': str(details.get('category', 'Unknown')),
                    'rating': float(details.get('rating', 0.0)),
                    'similarity': float(similarity),
                    'is_above_threshold': bool(similarity >= threshold),
                    'has_images': bool(len(details.get('image_paths', [])) > 0),
                    'image_count': int(details.get('image_count', 0)),
                    'image_paths': list(details.get('image_paths', [])),
                    'first_image': str(details.get('first_image')) if details.get('first_image') else None
                }
                
                if 'latitude' in details and 'longitude' in details:
                    result_item['latitude'] = float(details['latitude'])
                    result_item['longitude'] = float(details['longitude'])
                
                if 'address' in details:
                    result_item['address'] = str(details['address'])
                
                results.append(result_item)
            
            # Nettoyer
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Réponse
            return jsonify({
                'success': True,
                'message': f'Found {len(results)} similar attractions',
                'count': int(len(results)),
                'results': results,
                'search_info': {
                    'image_filename': str(image_file.filename),
                    'feature_dimension': int(image_features.shape[0]),
                    'total_compared': int(len(attractions_features)),
                    'threshold_applied': float(threshold)
                },
                'timestamp': str(datetime.now().isoformat())
            })
            
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"✗ Error during search: {e}")
            return jsonify({'success': False, 'error': f'Search failed: {str(e)}'}), 500
            
    except Exception as e:
        print(f"✗ Server error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/recommend/user/<int:user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """
    Recommandations personnalisées pour un utilisateur
    
    Paramètres GET:
        k: Nombre de recommandations (default: 10)
        use_cnn: Utiliser CNN (true/false, default: true)
        exclude_rated: Exclure les attractions notées (true/false, default: true)
    """
    if not system_initialized:
        return jsonify({'error': 'System not initialized', 'success': False}), 503
    
    try:
        k = int(request.args.get('k', 10))
        use_cnn = request.args.get('use_cnn', 'true').lower() == 'true'
        exclude_rated = request.args.get('exclude_rated', 'true').lower() == 'true'
        
        # Vérifier si l'utilisateur existe
        if user_id >= system_components['data_loader'].num_users:
            return jsonify({'error': f'User {user_id} does not exist', 'success': False}), 404
        
        print(f"Getting recommendations for user {user_id}, k={k}, use_cnn={use_cnn}")
        
        # Obtenir des recommandations
        recommendations = system_components['hybrid_recommender'].recommend_for_user(
            user_id=user_id,
            k=k,
            use_cnn=use_cnn,
            exclude_rated=exclude_rated
        )
        
        # Formater les résultats
        formatted_recs = []
        for att_id, score, rec_type in recommendations:
            details = get_attraction_details(att_id)
            
            formatted_recs.append({
                'attraction_id': int(att_id),
                'score': float(score),
                'type': str(rec_type),
                'details': details
            })
        
        # Obtenir l'historique de l'utilisateur
        user_history = system_components['data_loader'].get_user_history(user_id)
        history_data = []
        if not user_history.empty:
            for _, row in user_history.head(5).iterrows():
                att_details = get_attraction_details(int(row['attraction_id']))
                history_data.append({
                    'attraction_id': int(row['attraction_id']),
                    'rating': float(row['rating']),
                    'details': att_details
                })
        
        return jsonify({
            'user_id': int(user_id),
            'recommendations_count': int(len(formatted_recs)),
            'recommendations': formatted_recs,
            'user_history': history_data,
            'history_count': int(len(history_data)),
            'parameters': {
                'k': int(k),
                'use_cnn': bool(use_cnn),
                'exclude_rated': bool(exclude_rated)
            },
            'timestamp': str(datetime.now().isoformat())
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/recommend/cold-start', methods=['GET'])
def cold_start_recommendations():
    """
    Recommandations pour les nouveaux utilisateurs (démarrage à froid)
    
    Paramètres GET:
        k: Nombre de recommandations (default: 10)
        strategy: Stratégie (popular, trending, diverse, visual, hybrid, default: hybrid)
    """
    if not system_initialized:
        return jsonify({'error': 'System not initialized', 'success': False}), 503
    
    try:
        k = int(request.args.get('k', 10))
        strategy = request.args.get('strategy', 'hybrid')
        
        print(f"Getting cold-start recommendations, k={k}, strategy={strategy}")
        
        # Obtenir des recommandations
        recommendations = cold_start_solver.solve_cold_start(
            user_id=None,
            k=k,
            strategy=strategy
        )
        
        # Formater les résultats
        formatted_recs = []
        for att_id, score, details, rec_strategy in recommendations:
            attraction_details = get_attraction_details(att_id)
            details.update(attraction_details)
            
            formatted_recs.append({
                'attraction_id': int(att_id),
                'score': float(score),
                'strategy': str(rec_strategy),
                'details': details
            })
        
        return jsonify({
            'recommendations': formatted_recs,
            'parameters': {
                'k': int(k),
                'strategy': str(strategy)
            },
            'timestamp': str(datetime.now().isoformat())
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/recommend/hybrid', methods=['GET'])
def hybrid_recommendations():
    """
    Recommandations hybrides avancées
    
    Paramètres GET:
        user_id: ID utilisateur (requis)
        k: nombre de recommandations (default: 15)
        cnn_weight: poids des features CNN (0.0-1.0, default: 0.7)
        ncf_weight: poids du modèle NCF (0.0-1.0, default: 0.3)
        diversity: facteur de diversité (0.0-1.0, default: 0.2)
    """
    if not system_initialized:
        return jsonify({'error': 'System not initialized', 'success': False}), 503
    
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id parameter is required', 'success': False}), 400
        
        user_id = int(user_id)
        k = min(int(request.args.get('k', 15)), 50)
        cnn_weight = float(request.args.get('cnn_weight', 0.7))
        ncf_weight = float(request.args.get('ncf_weight', 0.3))
        diversity = float(request.args.get('diversity', 0.2))
        
        # Normaliser les poids
        total_weight = cnn_weight + ncf_weight
        if total_weight > 0:
            cnn_weight = cnn_weight / total_weight
            ncf_weight = ncf_weight / total_weight
        
        print(f"Hybrid recommendations for user {user_id}, k={k}, weights: CNN={cnn_weight:.2f}, NCF={ncf_weight:.2f}")
        
        # Vérifier si l'utilisateur existe
        if user_id >= system_components['data_loader'].num_users:
            return jsonify({'error': f'User {user_id} does not exist', 'success': False}), 404
        
        # Obtenir l'historique de l'utilisateur
        user_history = system_components['data_loader'].get_user_history(user_id)
        
        if user_history.empty:
            # Démarrage à froid
            cold_start_recs = cold_start_solver.solve_cold_start(
                user_id=user_id,
                k=k,
                strategy='hybrid'
            )
            
            formatted_recs = []
            for att_id, score, details, rec_strategy in cold_start_recs:
                attraction_details = get_attraction_details(att_id)
                details.update(attraction_details)
                
                formatted_recs.append({
                    'attraction_id': int(att_id),
                    'score': float(score),
                    'strategy': str(rec_strategy),
                    'type': 'cold_start',
                    'details': attraction_details
                })
            
            result_type = "cold_start"
            
        else:
            # Utilisateur existant - approche hybride simplifiée
            # Utiliser le hybrid_recommender existant avec différents poids
            hybrid_recommender = system_components['hybrid_recommender']
            
            # Temporairement ajuster alpha dans le hybrid_recommender
            original_alpha = getattr(hybrid_recommender, 'alpha', 0.5)
            
            # Définir alpha basé sur les poids
            hybrid_alpha = ncf_weight  # alpha = poids NCF
            
            try:
                hybrid_recommender.alpha = hybrid_alpha
                recommendations = hybrid_recommender.recommend_for_user(
                    user_id=user_id,
                    k=k * 2,  # Prendre plus pour la diversité
                    use_cnn=True,
                    exclude_rated=True
                )
            finally:
                # Restaurer l'alpha original
                hybrid_recommender.alpha = original_alpha
            
            # Appliquer la diversité
            final_recommendations = []
            categories_selected = set()
            
            for att_id, score, rec_type in recommendations:
                if len(final_recommendations) >= k:
                    break
                    
                details = get_attraction_details(att_id)
                category = details.get('category', 'Unknown')
                
                # Vérifier la diversité
                if diversity > 0 and len(categories_selected) > 0:
                    if category in categories_selected and np.random.random() < diversity:
                        continue
                
                categories_selected.add(category)
                final_recommendations.append((att_id, score, rec_type))
            
            # Formater les résultats
            formatted_recs = []
            for att_id, total_score, rec_type in final_recommendations:
                details = get_attraction_details(att_id)
                
                formatted_recs.append({
                    'attraction_id': int(att_id),
                    'total_score': float(total_score),
                    'details': details,
                    'type': str(rec_type),
                    'weight_ncf': float(ncf_weight),
                    'weight_cnn': float(cnn_weight)
                })
            
            result_type = "hybrid"
        
        # Statistiques
        stats = {
            'user_history_count': int(len(user_history)) if not user_history.empty else 0,
            'average_rating': float(user_history['rating'].mean()) if not user_history.empty else 0,
            'recommendation_type': str(result_type)
        }
        
        return jsonify({
            'success': True,
            'user_id': int(user_id),
            'recommendations': formatted_recs,
            'statistics': stats,
            'parameters': {
                'k': int(k),
                'cnn_weight': float(cnn_weight),
                'ncf_weight': float(ncf_weight),
                'diversity': float(diversity)
            },
            'timestamp': str(datetime.now().isoformat())
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/recommend/popular', methods=['GET'])
def popular_attractions():
    """
    Obtenir les attractions les plus populaires
    
    Paramètres GET:
        k: Nombre de recommandations (default: 10)
        category: Filtrer par catégorie (optionnel)
        min_reviews: Nombre minimum de reviews (default: 2)
    """
    if not system_initialized:
        return jsonify({'error': 'System not initialized', 'success': False}), 503
    
    try:
        k = int(request.args.get('k', 10))
        category = request.args.get('category', None)
        min_reviews = int(request.args.get('min_reviews', 2))
        
        print(f"Getting popular attractions, k={k}, category={category}")
        
        # Obtenir les attractions populaires
        popular = cold_start_solver.get_popular_attractions(
            k=k,
            min_reviews=min_reviews,
            category=category
        )
        
        # Formater les résultats
        formatted_popular = []
        for att_id, score, details in popular:
            attraction_details = get_attraction_details(att_id)
            details.update(attraction_details)
            
            formatted_popular.append({
                'attraction_id': int(att_id),
                'popularity_score': float(score),
                'details': details
            })
        
        return jsonify({
            'attractions': formatted_popular,
            'parameters': {
                'k': int(k),
                'category': str(category) if category else None,
                'min_reviews': int(min_reviews)
            },
            'timestamp': str(datetime.now().isoformat())
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/recommend/similar/<int:attraction_id>', methods=['GET'])
def similar_attractions(attraction_id):
    """
    Trouver des attractions similaires
    
    Paramètres GET:
        k: Nombre de recommandations (default: 10)
        similarity_type: Type de similarité (cnn, hybrid, default: cnn)
    """
    if not system_initialized:
        return jsonify({'error': 'System not initialized', 'success': False}), 503
    
    try:
        k = int(request.args.get('k', 10))
        similarity_type = request.args.get('similarity_type', 'cnn')
        
        # Vérifier si l'attraction existe
        if attraction_id >= system_components['data_loader'].num_attractions:
            return jsonify({'error': f'Attraction {attraction_id} does not exist', 'success': False}), 404
        
        print(f"Finding similar attractions for {attraction_id}, k={k}, type={similarity_type}")
        
        # Obtenir les attractions similaires
        if similarity_type == 'cnn':
            if attraction_id in system_components['attractions_features']:
                target_features = system_components['attractions_features'][attraction_id]
                similar = system_components['cnn_extractor'].find_similar_attractions(
                    target_features,
                    system_components['attractions_features'],
                    k=k
                )
            else:
                similar = []
        else:
            # Pour l'instant, utiliser CNN seulement pour hybrid
            if attraction_id in system_components['attractions_features']:
                target_features = system_components['attractions_features'][attraction_id]
                similar = system_components['cnn_extractor'].find_similar_attractions(
                    target_features,
                    system_components['attractions_features'],
                    k=k
                )
            else:
                similar = []
        
        # Formater les résultats
        formatted_similar = []
        for sim_id, score in similar:
            details = get_attraction_details(sim_id)
            
            formatted_similar.append({
                'attraction_id': int(sim_id),
                'similarity_score': float(score),
                'details': details
            })
        
        return jsonify({
            'source_attraction': {
                'id': int(attraction_id),
                'details': get_attraction_details(attraction_id)
            },
            'similar_attractions': formatted_similar,
            'similarity_type': str(similarity_type),
            'parameters': {
                'k': int(k),
                'similarity_type': str(similarity_type)
            },
            'timestamp': str(datetime.now().isoformat())
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/user/history/<int:user_id>', methods=['GET'])
def user_history(user_id):
    """
    Obtenir l'historique d'un utilisateur
    """
    if not system_initialized:
        return jsonify({'error': 'System not initialized', 'success': False}), 503
    
    try:
        # Vérifier si l'utilisateur existe
        if user_id >= system_components['data_loader'].num_users:
            return jsonify({'error': f'User {user_id} does not exist', 'success': False}), 404
        
        print(f"Getting history for user {user_id}")
        
        # Obtenir l'historique
        history_df = system_components['data_loader'].get_user_history(user_id)
        
        # Formater les résultats
        formatted_history = []
        if not history_df.empty:
            for _, row in history_df.iterrows():
                att_id = int(row['attraction_id'])
                details = get_attraction_details(att_id)
                
                formatted_history.append({
                    'attraction_id': att_id,
                    'rating': float(row['rating']),
                    'normalized_rating': float(row['rating_norm']),
                    'details': details
                })
        
        return jsonify({
            'user_id': int(user_id),
            'history': formatted_history,
            'total_rated': int(len(formatted_history)),
            'average_rating': float(history_df['rating'].mean()) if not history_df.empty else 0,
            'timestamp': str(datetime.now().isoformat())
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/attraction/info/<int:attraction_id>', methods=['GET'])
def attraction_info(attraction_id):
    """
    Obtenir les informations d'une attraction
    """
    if not system_initialized:
        return jsonify({'error': 'System not initialized', 'success': False}), 503
    
    try:
        # Vérifier si l'attraction existe
        if attraction_id >= system_components['data_loader'].num_attractions:
            return jsonify({'error': f'Attraction {attraction_id} does not exist', 'success': False}), 404
        
        print(f"Getting info for attraction {attraction_id}")
        
        details = get_attraction_details(attraction_id)
        
        # Ajouter les statistiques
        if system_components['data_loader'].interactions is not None:
            attraction_interactions = system_components['data_loader'].interactions[
                system_components['data_loader'].interactions['attraction_id'] == attraction_id
            ]
            
            if len(attraction_interactions) > 0:
                details['statistics'] = {
                    'review_count': int(len(attraction_interactions)),
                    'average_rating': float(attraction_interactions['rating'].mean()),
                    'rating_std': float(attraction_interactions['rating'].std()) if len(attraction_interactions) > 1 else 0
                }
        
        return jsonify(details)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/system/status', methods=['GET'])
def system_status():
    """
    Obtenir le statut du système
    """
    if system_initialized and system_components is not None:
        try:
            status = {
                'status': 'initialized',
                'components': {
                    'data_loader': {
                        'num_users': int(system_components['data_loader'].num_users),
                        'num_attractions': int(system_components['data_loader'].num_attractions),
                        'total_interactions': int(len(system_components['data_loader'].interactions)) if system_components['data_loader'].interactions is not None else 0
                    },
                    'ncf_model': {
                        'status': 'loaded' if system_components['ncf_model'] is not None else 'missing'
                    },
                    'cnn_extractor': {
                        'status': 'loaded' if system_components['cnn_extractor'] is not None else 'missing',
                        'attractions_features': int(len(system_components['attractions_features'])) if system_components['attractions_features'] is not None else 0
                    },
                    'cold_start_solver': cold_start_solver is not None
                },
                'timestamp': str(datetime.now().isoformat())
            }
            
            return jsonify(status)
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)})
    else:
        return jsonify({'status': 'not_initialized', 'message': 'System needs initialization'})

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        'status': 'healthy' if system_initialized else 'initializing',
        'system_initialized': bool(system_initialized),
        'timestamp': str(datetime.now().isoformat())
    })

if __name__ == '__main__':
    print("=" * 70)
    print("Starting Tourism Recommendation System API v2.0")
    print("=" * 70)
    
    # Initialiser le système
    if init_system():
        print("\n" + "=" * 70)
        print("API Server Ready!")
        print("Available at: http://localhost:5000")
        print("\nMain endpoints:")
        print("  GET  /                             - API homepage")
        print("  POST /search/by-image              - Search by image upload")
        print("  GET  /recommend/user/<id>          - Personalized recommendations")
        print("  GET  /recommend/cold-start         - Cold start recommendations")
        print("  GET  /recommend/hybrid             - Hybrid recommendations")
        print("  GET  /recommend/popular            - Popular attractions")
        print("  GET  /recommend/similar/<id>       - Similar attractions")
        print("  GET  /user/history/<id>            - User history")
        print("  GET  /attraction/info/<id>         - Attraction info")
        print("  GET  /system/status                - System status")
        print("  GET  /health                       - Health check")
        print("=" * 70 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "=" * 70)
        print("FAILED TO INITIALIZE SYSTEM!")
        print("=" * 70)