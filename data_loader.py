# data_loader.py
# Chargement des données (sans nettoyage supplémentaire)
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class TourismDataLoader:
    """
    Classe pour charger les données touristiques de Marrakech
    """
    def __init__(self, base_path):
        """
        Initialise le chargeur de données

        Args:
            base_path: Chemin de base vers les données
        """
        self.base_path = base_path
        self.reviews_path = os.path.join(base_path, "marrakech_reviews_clean_final.csv")
        self.attractions_path = os.path.join(base_path, "marrakech_attractions_clean_final.csv")
        self.images_base_path = os.path.join(base_path, "attractions_images")

        # DataFrames
        self.reviews_df = None
        self.attractions_df = None

        # Mappings
        self.user_encoder = LabelEncoder()
        self.attraction_encoder = LabelEncoder()

        # Données numériques
        self.num_users = 0
        self.num_attractions = 0
        self.interactions = None

    def load_data(self):
        """
        Charge les données depuis les fichiers CSV
        """
        print("Loading data (no additional cleaning)...")

        # Charger les données
        self.attractions_df = pd.read_csv(self.attractions_path)
        self.reviews_df = pd.read_csv(self.reviews_path)

        # Afficher les informations de base
        print(f"Attractions loaded: {len(self.attractions_df)}")
        print(f"Reviews loaded: {len(self.reviews_df)}")

        # SUPPRIMÉ: Nettoyer les données
        # self._clean_data()

        # Encoder les utilisateurs et attractions
        self._encode_ids()

        # Créer les interactions
        self._create_interactions()

        return self.reviews_df, self.attractions_df

    # SUPPRIMER COMPLÈTEMENT la méthode _clean_data
    # def _clean_data(self):
    #    ... # Tout supprimer ici

    def _encode_ids(self):
        """
        Encoder les IDs utilisateurs et attractions en indices numériques
        """
        # Encoder les URLs d'attractions (TOUTES les attractions du fichier)
        self.attraction_encoder.fit(self.attractions_df['attraction_url'].unique())

        # Encoder les noms d'utilisateurs (TOUS les utilisateurs des reviews)
        self.user_encoder.fit(self.reviews_df['reviewer_name'].unique())

        # Ajouter les IDs encodés aux DataFrames
        self.reviews_df['user_id'] = self.user_encoder.transform(self.reviews_df['reviewer_name'])
        self.reviews_df['attraction_id'] = self.attraction_encoder.transform(self.reviews_df['attraction_url'])

        self.attractions_df['attraction_id'] = self.attraction_encoder.transform(self.attractions_df['attraction_url'])

        # Mettre à jour les comptes
        self.num_users = len(self.user_encoder.classes_)
        self.num_attractions = len(self.attraction_encoder.classes_)

        print(f"Number of unique users: {self.num_users}")
        print(f"Number of unique attractions: {self.num_attractions}")
        print(f"All attractions kept (no filtering)")

    def _create_interactions(self):
        """
        Créer la matrice d'interactions utilisateur-attraction
        """
        # Agréger les ratings par utilisateur-attraction (moyenne)
        interactions = self.reviews_df.groupby(['user_id', 'attraction_id'])['rating'].mean().reset_index()

        # Normaliser les ratings entre 0 et 1
        interactions['rating_norm'] = (interactions['rating'] - 1) / 4

        self.interactions = interactions

        print(f"Total interactions: {len(self.interactions)}")

        return self.interactions

    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Diviser les données en train et test

        Args:
            test_size: Proportion des données de test
            random_state: Seed pour la reproductibilité

        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        if self.interactions is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Préparer les features et labels
        X = self.interactions[['user_id', 'attraction_id']].values
        y = self.interactions['rating_norm'].values

        # Split stratifié par utilisateur
        train_indices, test_indices = [], []

        for user_id in range(self.num_users):
            user_indices = np.where(X[:, 0] == user_id)[0]
            if len(user_indices) > 1:
                user_train, user_test = train_test_split(
                    user_indices,
                    test_size=test_size,
                    random_state=random_state
                )
                train_indices.extend(user_train)
                test_indices.extend(user_test)
            elif len(user_indices) == 1:
                train_indices.extend(user_indices)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        print(f"Train set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def get_user_history(self, user_id):
        """
        Obtenir l'historique des interactions d'un utilisateur

        Args:
            user_id: ID numérique de l'utilisateur

        Returns:
            DataFrame avec l'historique
        """
        if self.interactions is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.interactions[self.interactions['user_id'] == user_id]

    def get_attraction_info(self, attraction_id):
        """
        Obtenir les informations d'une attraction

        Args:
            attraction_id: ID numérique de l'attraction

        Returns:
            Series avec les informations
        """
        attraction_data = self.attractions_df[self.attractions_df['attraction_id'] == attraction_id]
        if len(attraction_data) > 0:
            return attraction_data.iloc[0]
        else:
            # Retourner une série vide si l'attraction n'est pas trouvée
            return pd.Series({'attraction_url': f'Unknown_{attraction_id}', 'category': 'Unknown', 'rating': 0})

    def get_attraction_image_paths(self, attraction_id):
        """
        Obtenir les chemins des images d'une attraction - VERSION CORRIGÉE
        """
        try:
            # Obtenir les informations de l'attraction
            attraction_row = self.attractions_df[
                self.attractions_df['attraction_id'] == attraction_id
            ].iloc[0]

            # Essayer d'abord avec le champ images_folder
            if 'images_folder' in attraction_row and pd.notna(attraction_row['images_folder']):
                images_folder = attraction_row['images_folder']
                if images_folder:
                    # Nettoyer le chemin du dossier
                    images_folder = images_folder.strip().rstrip('/')
                    full_folder_path = os.path.join(self.images_base_path, images_folder)

                    if os.path.exists(full_folder_path):
                        # Lire les images du dossier
                        image_files = []
                        for file in sorted(os.listdir(full_folder_path)):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                                full_path = os.path.join(full_folder_path, file)
                                if os.path.exists(full_path):
                                    image_files.append(full_path)

                        if image_files:
                            return image_files[:5]

            # Si pas trouvé via images_folder, essayer avec le nom de l'attraction
            attraction_name = attraction_row.get('attraction_name', '')
            if not attraction_name:
                # Extraire le nom de l'URL
                attraction_url = attraction_row.get('attraction_url', '')
                if 'Reviews-' in attraction_url:
                    attraction_name = attraction_url.split('Reviews-')[1].replace('.html', '')
                else:
                    attraction_name = os.path.basename(attraction_url).replace('.html', '')

            # Nettoyer le nom pour la recherche
            clean_name = self._clean_attraction_name(attraction_name)

            # Chercher récursivement dans le dossier images
            image_paths = []
            for root, dirs, files in os.walk(self.images_base_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                        file_lower = file.lower()
                        clean_name_lower = clean_name.lower()

                        # Vérifier différentes correspondances
                        if (clean_name_lower in file_lower or
                            clean_name_lower.replace(' ', '-') in file_lower or
                            clean_name_lower.replace(' ', '_') in file_lower or
                            str(attraction_id) in file_lower):

                            full_path = os.path.join(root, file)
                            if os.path.exists(full_path):
                                image_paths.append(full_path)

            # Retourner les 5 premières images uniques
            unique_paths = []
            seen = set()
            for path in sorted(image_paths):
                if path not in seen:
                    seen.add(path)
                    unique_paths.append(path)

            return unique_paths[:5]

        except Exception as e:
            print(f"Warning: Error getting images for attraction {attraction_id}: {str(e)[:100]}")
            return []

    def check_image_availability(self, n_samples=10):
        """
        Vérifier la disponibilité des images
        """
        print("Checking image availability...")

        results = []
        sample_attractions = self.attractions_df.head(n_samples)

        for _, row in sample_attractions.iterrows():
            attraction_id = row['attraction_id']
            attraction_name = row.get('attraction_name', 'Unknown')
            images_folder = row.get('images_folder', '')

            # Obtenir les chemins d'images
            image_paths = self.get_attraction_image_paths(attraction_id)

            # Vérifier le dossier spécifié
            folder_exists = False
            if images_folder:
                folder_path = os.path.join(self.images_base_path, images_folder.strip())
                folder_exists = os.path.exists(folder_path)

            results.append({
                'attraction_id': attraction_id,
                'name': attraction_name[:40],
                'images_folder': images_folder,
                'folder_exists': folder_exists,
                'has_images': len(image_paths) > 0,
                'image_count': len(image_paths),
                'sample_path': os.path.basename(image_paths[0]) if image_paths else 'None'
            })

            # Afficher des détails pour le debug
            print(f"\nAttraction: {attraction_name[:40]}...")
            print(f"  ID: {attraction_id}")
            print(f"  Dossier configuré: {images_folder}")
            print(f"  Dossier existe: {folder_exists}")
            print(f"  Images trouvées: {len(image_paths)}")
            if image_paths:
                print(f"  Exemple: {os.path.basename(image_paths[0])}")

        results_df = pd.DataFrame(results)

        print(f"\nImage Availability Summary:")
        print(f"Total checked: {len(results_df)}")
        print(f"With images: {sum(results_df['has_images'])}")
        print(f"Without images: {len(results_df) - sum(results_df['has_images'])}")

        return results_df

    def get_interaction_matrix(self):
        """
        Créer une matrice d'interactions sparse

        Returns:
            Matrice d'interactions (users x attractions)
        """
        if self.interactions is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Créer une matrice sparse
        interaction_matrix = np.zeros((self.num_users, self.num_attractions))

        for _, row in self.interactions.iterrows():
            user_id = int(row['user_id'])
            attraction_id = int(row['attraction_id'])
            rating = row['rating_norm']
            interaction_matrix[user_id, attraction_id] = rating

        return interaction_matrix

    def _clean_attraction_name(self, name):
        """
        Nettoyer le nom d'une attraction pour la recherche
        """
        if not isinstance(name, str):
            return ""

        # Enlever les parties inutiles
        name = name.replace('_', ' ').replace('-', ' ').replace('.html', '')

        # Enlever les préfixes communs
        prefixes = ['Reviews-', 'Attraction_Review-']
        for prefix in prefixes:
            if prefix in name:
                name = name.split(prefix)[-1]

        # Nettoyer les espaces
        name = ' '.join(name.split())

        return name


    def analyze_data_distribution(self):
        """
        Analyser la distribution des données
        """
        print("\n" + "="*50)
        print("DATA DISTRIBUTION ANALYSIS")
        print("="*50)

        # Distribution des ratings
        if self.interactions is not None:
            ratings = self.interactions['rating']
            print(f"\n1. Rating Distribution:")
            print(f"   Min: {ratings.min():.1f}")
            print(f"   Max: {ratings.max():.1f}")
            print(f"   Mean: {ratings.mean():.2f}")
            print(f"   Std: {ratings.std():.2f}")

            # Distribution par utilisateur
            user_counts = self.interactions['user_id'].value_counts()
            print(f"\n2. User Activity:")
            print(f"   Total users: {self.num_users}")
            print(f"   Users with reviews: {len(user_counts)}")
            print(f"   Avg reviews per user: {user_counts.mean():.2f}")
            print(f"   Most active user: {user_counts.max()} reviews")

            # Distribution par attraction
            attraction_counts = self.interactions['attraction_id'].value_counts()
            print(f"\n3. Attraction Popularity:")
            print(f"   Total attractions: {self.num_attractions}")
            print(f"   Attractions with reviews: {len(attraction_counts)}")
            print(f"   Avg reviews per attraction: {attraction_counts.mean():.2f}")
            print(f"   Most popular attraction: {attraction_counts.max()} reviews")

            # Sparsity
            sparsity = 1 - (len(self.interactions) / (self.num_users * self.num_attractions))
            print(f"\n4. Matrix Statistics:")
            print(f"   Total possible interactions: {self.num_users * self.num_attractions}")
            print(f"   Actual interactions: {len(self.interactions)}")
            print(f"   Sparsity: {sparsity:.2%}")
            print(f"   Density: {1-sparsity:.2%}")


