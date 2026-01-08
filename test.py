# test.py - VERSION LOCALE
# Script de test complet pour le système de recommandation - VERSION LOCALE
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

class RecommendationSystemTester:
    """
    Classe pour tester le système de recommandation - VERSION LOCALE
    """

    def __init__(self, system_components):
        """
        Initialiser le testeur avec les composants du système
        """
        self.data_loader = system_components['data_loader']
        self.ncf_model = system_components['ncf_model']
        self.cnn_extractor = system_components['cnn_extractor']
        self.hybrid_recommender = system_components['hybrid_recommender']
        self.evaluator = system_components['evaluator']
        self.attractions_features = system_components['attractions_features']

        print("=" * 70)
        print("SYSTEME DE TEST INITIALISE - VERSION LOCALE")
        print("=" * 70)
        print(f"Utilisateurs: {self.data_loader.num_users}")
        print(f"Attractions: {self.data_loader.num_attractions}")
        print(f"Features CNN: {len(self.attractions_features)}")
        print("=" * 70)

    def display_attraction_with_image(self, attraction_id, show_image=True):
        """
        Afficher les détails d'une attraction avec son image
        """
        try:
            info = self.data_loader.get_attraction_info(attraction_id)

            # Nettoyer le nom
            name = info.get('attraction_url', 'Unknown')
            if isinstance(name, str):
                if 'Reviews-' in name:
                    name = name.split('Reviews-')[1]
                if '.html' in name:
                    name = name.replace('.html', '')
                name = name.replace('_', ' ').replace('-', ' ').title()

            category = info.get('category', 'Non specifie')
            if pd.isna(category):
                category = 'Non specifie'

            rating = info.get('rating', 'N/A')
            if pd.isna(rating):
                rating = 'N/A'

            print(f"  {name}")
            print(f"     Categorie: {category}")
            print(f"     Note: {rating}")

            # Afficher l'image si demande
            if show_image:
                image_paths = self.data_loader.get_attraction_image_paths(attraction_id)
                if image_paths:
                    try:
                        img = Image.open(image_paths[0])
                        img.thumbnail((200, 200))
                        plt.figure(figsize=(3, 3))
                        plt.imshow(img)
                        plt.axis('off')
                        plt.title(name[:30])
                        plt.show()
                    except Exception as e:
                        print(f"     Erreur chargement image: {e}")
                else:
                    print(f"     Pas d'image disponible")

        except Exception as e:
            print(f"  Erreur affichage attraction {attraction_id}: {e}")

    # ==================== TEST NCF ====================

    def test_ncf_existing_user(self):
        """
        TEST NCF Option 1: Predictions pour un utilisateur existant avec comparaison
        """
        print("\n" + "=" * 70)
        print("TEST NCF - OPTION 1: UTILISATEUR EXISTANT")
        print("=" * 70)

        # Trouver des utilisateurs avec historique
        user_counts = self.data_loader.interactions['user_id'].value_counts()
        users_with_history = user_counts[user_counts >= 5].index.tolist()

        if not users_with_history:
            print("Aucun utilisateur avec suffisamment d'historique trouve")
            return

        # Selectionner un utilisateur aleatoire
        test_user = random.choice(users_with_history)

        print(f"\nUtilisateur selectionne: {test_user}")

        # Obtenir l'historique
        user_history = self.data_loader.get_user_history(test_user)

        print(f"Nombre d'attractions notees: {len(user_history)}")
        print(f"Note moyenne: {user_history['rating'].mean():.2f}/5.0")

        # Afficher quelques attractions notees
        print(f"\n{'='*70}")
        print("HISTORIQUE DE L'UTILISATEUR (echantillon)")
        print("="*70)

        sample_history = user_history.sample(min(5, len(user_history)))

        for idx, (_, row) in enumerate(sample_history.iterrows(), 1):
            attraction_id = row['attraction_id']
            actual_rating = row['rating']

            # Predire le rating
            predicted_rating = self.ncf_model.predict(test_user, attraction_id)
            predicted_rating_scaled = predicted_rating * 4 + 1

            error = abs(actual_rating - predicted_rating_scaled)

            print(f"\n{idx}. Attraction ID: {attraction_id}")
            self.display_attraction_with_image(attraction_id, show_image=False)
            print(f"     Note reelle: {actual_rating:.2f}/5.0")
            print(f"     Note predite: {predicted_rating_scaled:.2f}/5.0")
            print(f"     Erreur: {error:.2f}")

        # Obtenir des recommandations pour nouvelles attractions
        print(f"\n{'='*70}")
        print("RECOMMANDATIONS POUR NOUVELLES ATTRACTIONS")
        print("="*70)

        recommendations = self.ncf_model.recommend_top_k(
            user_id=test_user,
            k=10,
            exclude_rated=True,
            data_loader=self.data_loader
        )

        if recommendations:
            print(f"\nTop 10 attractions recommandees:")

            results_table = []
            for rank, (att_id, score) in enumerate(recommendations, 1):
                info = self.data_loader.get_attraction_info(att_id)

                name = info.get('attraction_url', 'Unknown')
                if isinstance(name, str):
                    if 'Reviews-' in name:
                        name = name.split('Reviews-')[1]
                    if '.html' in name:
                        name = name.replace('.html', '')
                    name = name.replace('_', ' ').replace('-', ' ').title()

                category = info.get('category', 'Non specifie')
                if pd.isna(category):
                    category = 'Non specifie'

                rating = info.get('rating', 'N/A')

                results_table.append([
                    rank,
                    name[:40],
                    category[:25],
                    rating,
                    f"{score:.4f}"
                ])

            headers = ["Rang", "Nom", "Categorie", "Note", "Score NCF"]
            print(tabulate(results_table, headers=headers, tablefmt="grid"))
        else:
            print("Aucune recommandation disponible")

    def test_ncf_new_attractions(self):
        """
        TEST NCF Option 2: Recommandations pour attractions non notees
        """
        print("\n" + "=" * 70)
        print("TEST NCF - OPTION 2: NOUVELLES ATTRACTIONS")
        print("=" * 70)

        # Selectionner un utilisateur
        user_counts = self.data_loader.interactions['user_id'].value_counts()
        users_with_history = user_counts[user_counts >= 3].index.tolist()

        if not users_with_history:
            print("Aucun utilisateur avec historique trouve")
            return

        test_user = random.choice(users_with_history)

        print(f"\nUtilisateur selectionne: {test_user}")

        # Obtenir les recommandations
        recommendations = self.ncf_model.recommend_top_k(
            user_id=test_user,
            k=10,
            exclude_rated=True,
            data_loader=self.data_loader
        )

        if not recommendations:
            print("Aucune recommandation disponible")
            return

        print(f"\nTop 10 attractions recommandees (non notees par l'utilisateur):")

        results_table = []
        for rank, (att_id, score) in enumerate(recommendations, 1):
            info = self.data_loader.get_attraction_info(att_id)

            name = info.get('attraction_url', 'Unknown')
            if isinstance(name, str):
                if 'Reviews-' in name:
                    name = name.split('Reviews-')[1]
                if '.html' in name:
                    name = name.replace('.html', '')
                name = name.replace('_', ' ').replace('-', ' ').title()

            category = info.get('category', 'Non specifie')
            if pd.isna(category):
                category = 'Non specifie'

            rating = info.get('rating', 'N/A')

            has_images = len(self.data_loader.get_attraction_image_paths(att_id)) > 0

            results_table.append([
                rank,
                name[:40],
                category[:25],
                f"{rating}",
                f"{score:.4f}",
                "Oui" if has_images else "Non"
            ])

        headers = ["Rang", "Nom", "Categorie", "Note", "Score", "Images"]
        print(tabulate(results_table, headers=headers, tablefmt="grid"))

        # Afficher quelques images
        print(f"\n{'='*70}")
        print("APERCU DES IMAGES (Top 3)")
        print("="*70)

        for rank, (att_id, score) in enumerate(recommendations[:3], 1):
            print(f"\n{rank}. Score: {score:.4f}")
            self.display_attraction_with_image(att_id, show_image=True)

    # ==================== TEST CNN ====================

    def test_cnn_random_image(self):
        """
        TEST CNN Option 1: Image aleatoire et top 5 attractions similaires
        """
        print("\n" + "=" * 70)
        print("TEST CNN - OPTION 1: IMAGE ALEATOIRE")
        print("=" * 70)

        # Selectionner une attraction aleatoire avec images
        attractions_with_images = []

        for att_id in list(self.attractions_features.keys())[:100]:
            image_paths = self.data_loader.get_attraction_image_paths(att_id)
            if image_paths:
                attractions_with_images.append((att_id, image_paths))

        if not attractions_with_images:
            print("Aucune attraction avec images trouvee")
            return

        # Choisir une attraction aleatoire
        selected_att_id, image_paths = random.choice(attractions_with_images)
        selected_image = random.choice(image_paths)

        print(f"\nImage selectionnee:")
        print(f"Attraction ID: {selected_att_id}")
        print(f"Chemin: {os.path.basename(selected_image)}")

        # Afficher l'image
        try:
            img = Image.open(selected_image)
            img.thumbnail((300, 300))
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.axis('off')
            plt.title("Image de requete")
            plt.show()
        except Exception as e:
            print(f"Erreur affichage image: {e}")

        # Trouver les attractions similaires
        print(f"\n{'='*70}")
        print("TOP 5 ATTRACTIONS SIMILAIRES")
        print("="*70)

        if selected_att_id in self.attractions_features:
            target_features = self.attractions_features[selected_att_id]

            similar_attractions = self.cnn_extractor.find_similar_attractions(
                target_features=target_features,
                features_dict=self.attractions_features,
                k=6  # 6 car le premier sera l'image elle-meme
            )

            # Exclure la premiere (l'image elle-meme)
            similar_attractions = similar_attractions[1:6]

            results_table = []
            for rank, (att_id, similarity) in enumerate(similar_attractions, 1):
                info = self.data_loader.get_attraction_info(att_id)

                name = info.get('attraction_url', 'Unknown')
                if isinstance(name, str):
                    if 'Reviews-' in name:
                        name = name.split('Reviews-')[1]
                    if '.html' in name:
                        name = name.replace('.html', '')
                    name = name.replace('_', ' ').replace('-', ' ').title()

                category = info.get('category', 'Non specifie')
                if pd.isna(category):
                    category = 'Non specifie'

                rating = info.get('rating', 'N/A')

                results_table.append([
                    rank,
                    name[:40],
                    category[:25],
                    f"{rating}",
                    f"{similarity:.4f}"
                ])

            headers = ["Rang", "Nom", "Categorie", "Note", "Similarite"]
            print(tabulate(results_table, headers=headers, tablefmt="grid"))

            # Afficher les images
            print(f"\n{'='*70}")
            print("IMAGES DES ATTRACTIONS SIMILAIRES")
            print("="*70)

            fig, axes = plt.subplots(1, 5, figsize=(15, 3))

            for idx, (att_id, similarity) in enumerate(similar_attractions):
                image_paths = self.data_loader.get_attraction_image_paths(att_id)

                if image_paths:
                    try:
                        img = Image.open(image_paths[0])
                        axes[idx].imshow(img)
                        axes[idx].axis('off')
                        axes[idx].set_title(f"#{idx+1}\nSim: {similarity:.3f}", fontsize=8)
                    except:
                        axes[idx].text(0.5, 0.5, 'Erreur', ha='center', va='center')
                        axes[idx].axis('off')
                else:
                    axes[idx].text(0.5, 0.5, 'Pas d\'image', ha='center', va='center')
                    axes[idx].axis('off')

            plt.tight_layout()
            plt.show()
        else:
            print("Features non disponibles pour cette attraction")

    def test_cnn_custom_image(self, image_path=None):
        """
        TEST CNN Option 2: Image personnalisee et top 5 attractions
        """
        print("\n" + "=" * 70)
        print("TEST CNN - OPTION 2: IMAGE PERSONNALISEE")
        print("=" * 70)

        if image_path is None:
            print("\nVeuillez fournir un chemin d'image.")
            print("Exemple: tester.test_cnn_custom_image('C:\\\\chemin\\\\vers\\\\image.jpg')")

            # Proposer une image par defaut
            print("\nCherche une image par defaut...")

            # Chercher une image dans le dossier d'images
            images_path = self.data_loader.images_base_path
            if os.path.exists(images_path):
                for root, dirs, files in os.walk(images_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(root, file)
                            print(f"Image trouvee: {image_path}")
                            break
                    if image_path:
                        break

            if image_path is None:
                print("Aucune image disponible")
                return

        if not os.path.exists(image_path):
            print(f"Image non trouvee: {image_path}")
            return

        print(f"\nImage selectionnee: {os.path.basename(image_path)}")

        # Afficher l'image
        try:
            img = Image.open(image_path)
            img.thumbnail((300, 300))
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.axis('off')
            plt.title("Votre image")
            plt.show()
        except Exception as e:
            print(f"Erreur affichage image: {e}")
            return

        # Extraire les features et trouver les similaires
        print(f"\n{'='*70}")
        print("RECHERCHE DES ATTRACTIONS SIMILAIRES...")
        print("="*70)

        try:
            # Essayer d'abord avec le vrai extracteur
            if hasattr(self.cnn_extractor, 'extract_features_batch'):
                # Vrai extracteur CNN
                similar_attractions = self.cnn_extractor.find_similar_by_image(
                    input_image_path=image_path,
                    features_dict=self.attractions_features,
                    k=5
                )
            else:
                # SimpleCNNExtractor - extraire features manuellement
                print("Utilisation du mode simplifie (features pre-calculees)")
                print("Extraction des features de votre image...")

                # Utiliser un extracteur temporaire pour cette image
                import torch
                import torchvision.models as models
                import torchvision.transforms as transforms

                # Charger le modele
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                model = torch.nn.Sequential(*list(model.children())[:-1])
                model.eval()
                model = model.to(device)

                # Transformations
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

                # Charger et transformer l'image
                img = Image.open(image_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Extraire les features
                with torch.no_grad():
                    features = model(img_tensor)
                    features = features.squeeze()
                    features = torch.nn.functional.normalize(features.unsqueeze(0), p=2, dim=1)
                    target_features = features.cpu().numpy()[0]

                # Nettoyer
                del model, img_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Trouver les similaires
                similar_attractions = self.cnn_extractor.find_similar_attractions(
                    target_features=target_features,
                    features_dict=self.attractions_features,
                    k=5
                )

            if not similar_attractions:
                print("Aucune attraction similaire trouvee")
                return

            print(f"\nTop 5 attractions similaires trouvees:")

            results_table = []
            for rank, (att_id, similarity) in enumerate(similar_attractions, 1):
                info = self.data_loader.get_attraction_info(att_id)

                name = info.get('attraction_url', 'Unknown')
                if isinstance(name, str):
                    if 'Reviews-' in name:
                        name = name.split('Reviews-')[1]
                    if '.html' in name:
                        name = name.replace('.html', '')
                    name = name.replace('_', ' ').replace('-', ' ').title()

                category = info.get('category', 'Non specifie')
                if pd.isna(category):
                    category = 'Non specifie'

                rating = info.get('rating', 'N/A')

                results_table.append([
                    rank,
                    name[:40],
                    category[:25],
                    f"{rating}",
                    f"{similarity:.4f}"
                ])

            headers = ["Rang", "Nom", "Categorie", "Note", "Similarite"]
            print(tabulate(results_table, headers=headers, tablefmt="grid"))

            # Afficher les images des resultats
            print(f"\n{'='*70}")
            print("IMAGES DES ATTRACTIONS TROUVEES")
            print("="*70)

            fig, axes = plt.subplots(1, 5, figsize=(15, 3))

            for idx, (att_id, similarity) in enumerate(similar_attractions):
                image_paths = self.data_loader.get_attraction_image_paths(att_id)

                if image_paths:
                    try:
                        img = Image.open(image_paths[0])
                        axes[idx].imshow(img)
                        axes[idx].axis('off')

                        info = self.data_loader.get_attraction_info(att_id)
                        name = info.get('attraction_url', 'Unknown')
                        if isinstance(name, str):
                            if 'Reviews-' in name:
                                name = name.split('Reviews-')[1]
                            if '.html' in name:
                                name = name.replace('.html', '')
                            name = name.replace('_', ' ').replace('-', ' ').title()

                        axes[idx].set_title(f"#{idx+1}: {name[:20]}\nSim: {similarity:.3f}",
                                          fontsize=8)
                    except:
                        axes[idx].text(0.5, 0.5, 'Erreur', ha='center', va='center')
                        axes[idx].axis('off')
                else:
                    axes[idx].text(0.5, 0.5, 'Pas d\'image', ha='center', va='center')
                    axes[idx].axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Erreur lors de la recherche: {e}")
            import traceback
            traceback.print_exc()

    # ==================== TEST HYBRIDE ====================

    def test_hybrid_recommendations(self, alpha_values=[0.0, 0.3, 0.5, 0.7, 1.0]):
        """
        TEST HYBRIDE: Tester differentes valeurs d'alpha (NCF vs CNN)
        """
        print("\n" + "=" * 70)
        print("TEST HYBRIDE - FUSION NCF ET CNN")
        print("=" * 70)

        # Selectionner un utilisateur
        user_counts = self.data_loader.interactions['user_id'].value_counts()
        users_with_history = user_counts[user_counts >= 5].index.tolist()

        if not users_with_history:
            print("Aucun utilisateur avec historique trouve")
            return

        test_user = random.choice(users_with_history)

        print(f"\nUtilisateur selectionne: {test_user}")

        user_history = self.data_loader.get_user_history(test_user)
        print(f"Historique: {len(user_history)} attractions notees")

        # Tester differentes valeurs d'alpha
        print(f"\n{'='*70}")
        print("COMPARAISON DES DIFFERENTES VALEURS D'ALPHA")
        print("="*70)
        print("\nAlpha = 0.0 -> 100% CNN (visuel)")
        print("Alpha = 0.5 -> 50% NCF + 50% CNN")
        print("Alpha = 1.0 -> 100% NCF (collaboratif)")

        all_results = {}

        for alpha in alpha_values:
            print(f"\n{'-'*70}")
            print(f"ALPHA = {alpha} ({int(alpha*100)}% NCF + {int((1-alpha)*100)}% CNN)")
            print(f"{'-'*70}")

            self.hybrid_recommender.set_alpha(alpha)

            recommendations = self.hybrid_recommender.recommend_for_user(
                user_id=test_user,
                k=5,
                use_cnn=True,
                exclude_rated=True,
                force_recompute=True
            )

            if not recommendations:
                print("Aucune recommandation disponible")
                continue

            all_results[alpha] = recommendations

            results_table = []
            for rank, (att_id, score, rec_type) in enumerate(recommendations, 1):
                info = self.data_loader.get_attraction_info(att_id)

                name = info.get('attraction_url', 'Unknown')
                if isinstance(name, str):
                    if 'Reviews-' in name:
                        name = name.split('Reviews-')[1]
                    if '.html' in name:
                        name = name.replace('.html', '')
                    name = name.replace('_', ' ').replace('-', ' ').title()

                category = info.get('category', 'Non specifie')
                if pd.isna(category):
                    category = 'Non specifie'

                rating = info.get('rating', 'N/A')

                results_table.append([
                    rank,
                    name[:35],
                    category[:20],
                    f"{rating}",
                    f"{score:.4f}",
                    rec_type
                ])

            headers = ["Rang", "Nom", "Categorie", "Note", "Score", "Type"]
            print(tabulate(results_table, headers=headers, tablefmt="grid"))

        # Analyse comparative
        print(f"\n{'='*70}")
        print("ANALYSE COMPARATIVE")
        print("="*70)

        if len(all_results) >= 2:
            # Comparer les recommendations
            comparison_data = []

            for alpha in alpha_values:
                if alpha in all_results:
                    recs = all_results[alpha]
                    avg_score = np.mean([score for _, score, _ in recs])

                    comparison_data.append([
                        f"{alpha}",
                        f"{int(alpha*100)}% NCF",
                        f"{int((1-alpha)*100)}% CNN",
                        f"{avg_score:.4f}",
                        f"{len(recs)}"
                    ])

            headers = ["Alpha", "Part NCF", "Part CNN", "Score Moyen", "Nb Recs"]
            print(tabulate(comparison_data, headers=headers, tablefmt="grid"))

            # Visualiser les scores
            plt.figure(figsize=(10, 5))

            alphas_plot = []
            scores_plot = []

            for alpha in sorted(all_results.keys()):
                recs = all_results[alpha]
                avg_score = np.mean([score for _, score, _ in recs])
                alphas_plot.append(alpha)
                scores_plot.append(avg_score)

            plt.plot(alphas_plot, scores_plot, marker='o', linewidth=2, markersize=8)
            plt.xlabel('Alpha (part de NCF)', fontsize=12)
            plt.ylabel('Score moyen des recommandations', fontsize=12)
            plt.title('Impact du parametre Alpha sur les scores', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(alphas_plot)
            plt.tight_layout()
            plt.show()

        # Afficher les images pour alpha = 0.5
        if 0.5 in all_results:
            print(f"\n{'='*70}")
            print("APERCU VISUEL (ALPHA = 0.5)")
            print("="*70)

            recommendations = all_results[0.5]

            fig, axes = plt.subplots(1, 5, figsize=(15, 3))

            for idx, (att_id, score, rec_type) in enumerate(recommendations):
                image_paths = self.data_loader.get_attraction_image_paths(att_id)

                if image_paths:
                    try:
                        img = Image.open(image_paths[0])
                        axes[idx].imshow(img)
                        axes[idx].axis('off')

                        info = self.data_loader.get_attraction_info(att_id)
                        name = info.get('attraction_url', 'Unknown')
                        if isinstance(name, str):
                            if 'Reviews-' in name:
                                name = name.split('Reviews-')[1]
                            if '.html' in name:
                                name = name.replace('.html', '')
                            name = name.replace('_', ' ').replace('-', ' ').title()

                        axes[idx].set_title(f"#{idx+1}: {name[:15]}\n{rec_type}",
                                          fontsize=8)
                    except:
                        axes[idx].text(0.5, 0.5, 'Erreur', ha='center', va='center')
                        axes[idx].axis('off')
                else:
                    axes[idx].text(0.5, 0.5, 'Pas d\'image', ha='center', va='center')
                    axes[idx].axis('off')

            plt.tight_layout()
            plt.show()


# ==================== FONCTION PRINCIPALE ====================

def run_all_tests(system_components):
    """
    Executer tous les tests du systeme
    """
    print("\n" + "=" * 70)
    print("DEMARRAGE DE LA SUITE DE TESTS COMPLETE - VERSION LOCALE")
    print("=" * 70)

    # Creer le testeur
    tester = RecommendationSystemTester(system_components)

    # Menu interactif
    while True:
        print("\n" + "=" * 70)
        print("MENU DES TESTS")
        print("=" * 70)
        print("\nTESTS NCF (Neural Collaborative Filtering):")
        print("  1. Test NCF - Utilisateur existant (predictions + comparaison)")
        print("  2. Test NCF - Nouvelles attractions (recommendations)")

        print("\nTESTS CNN (Recherche visuelle):")
        print("  3. Test CNN - Image aleatoire (top 5 similaires)")
        print("  4. Test CNN - Image personnalisee (votre image)")

        print("\nTESTS HYBRIDES:")
        print("  5. Test Hybride - Comparaison alpha (NCF + CNN)")

        print("\nAUTRES:")
        print("  6. Executer TOUS les tests")
        print("  0. Quitter")

        print("=" * 70)

        choice = input("\nVotre choix (0-6): ").strip()

        if choice == '1':
            tester.test_ncf_existing_user()

        elif choice == '2':
            tester.test_ncf_new_attractions()

        elif choice == '3':
            tester.test_cnn_random_image()

        elif choice == '4':
            image_path = input("Chemin de l'image (ou Enter pour image par defaut): ").strip()
            if not image_path:
                image_path = None
            tester.test_cnn_custom_image(image_path)

        elif choice == '5':
            tester.test_hybrid_recommendations()

        elif choice == '6':
            print("\nExecution de tous les tests...")
            tester.test_ncf_existing_user()
            input("\nAppuyez sur Enter pour continuer...")

            tester.test_ncf_new_attractions()
            input("\nAppuyez sur Enter pour continuer...")

            tester.test_cnn_random_image()
            input("\nAppuyez sur Enter pour continuer...")

            tester.test_hybrid_recommendations()

            print("\n" + "=" * 70)
            print("TOUS LES TESTS TERMINES")
            print("=" * 70)

        elif choice == '0':
            print("\nAu revoir!")
            break

        else:
            print("\nChoix invalide. Veuillez reessayer.")

    return tester


# ==================== EXECUTION ====================

if __name__ == "__main__":
    print("""
    =====================================================================
    SCRIPT DE TEST DU SYSTEME DE RECOMMANDATION - VERSION LOCALE
    =====================================================================

    Ce script doit etre execute APRES main.py

    Utilisation:

    1. Executez d'abord main.py pour obtenir 'system_components'

    2. Puis executez ce script:
       from test import run_all_tests
       run_all_tests(system_components)

    =====================================================================
    """)