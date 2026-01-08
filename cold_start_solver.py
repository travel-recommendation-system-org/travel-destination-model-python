# cold_start_solver.py
# Gestion du démarrage à froid pour les nouveaux utilisateurs
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ColdStartSolver:
    """
    Classe pour résoudre le problème de démarrage à froid
    Fournit des recommandations basées sur la popularité pour les nouveaux utilisateurs
    """
    
    def __init__(self, data_loader, attractions_features=None):
        """
        Initialise le solveur de démarrage à froid
        
        Args:
            data_loader: Instance de TourismDataLoader
            attractions_features: Dictionnaire des features CNN (optionnel)
        """
        self.data_loader = data_loader
        self.attractions_features = attractions_features
        
        # Cache pour les résultats
        self.popular_attractions_cache = None
        self.category_popular_cache = {}
        self.trending_attractions_cache = None
        
        # Compteurs pour l'analyse
        self.recommendation_counts = defaultdict(int)
        
        logger.info("ColdStartSolver initialisé")
    
    def calculate_popularity_scores(self, min_reviews=1, use_weighted=True):
        """
        Calcule les scores de popularité pour toutes les attractions
        
        Args:
            min_reviews: Nombre minimum de reviews pour considérer une attraction
            use_weighted: Utiliser un score pondéré (rating * nombre de reviews)
            
        Returns:
            DataFrame avec les scores de popularité
        """
        if self.data_loader.interactions is None:
            logger.error("Données d'interactions non disponibles")
            return pd.DataFrame()
        
        try:
            # Agrégation des données
            popularity_data = []
            
            for attraction_id in range(self.data_loader.num_attractions):
                # Obtenir les interactions pour cette attraction
                attraction_interactions = self.data_loader.interactions[
                    self.data_loader.interactions['attraction_id'] == attraction_id
                ]
                
                if len(attraction_interactions) >= min_reviews:
                    # Informations de base
                    review_count = len(attraction_interactions)
                    avg_rating = attraction_interactions['rating'].mean()
                    
                    # Calcul du score de popularité
                    if use_weighted:
                        # Score pondéré: moyenne des ratings * log(nombre de reviews + 1)
                        popularity_score = avg_rating * np.log1p(review_count)
                    else:
                        popularity_score = avg_rating
                    
                    # Informations supplémentaires
                    rating_std = attraction_interactions['rating'].std()
                    if pd.isna(rating_std):
                        rating_std = 0
                    
                    # Obtenir la catégorie
                    attraction_info = self.data_loader.get_attraction_info(attraction_id)
                    category = attraction_info.get('category', 'Unknown') if not attraction_info.empty else 'Unknown'
                    
                    popularity_data.append({
                        'attraction_id': attraction_id,
                        'review_count': review_count,
                        'avg_rating': avg_rating,
                        'rating_std': rating_std,
                        'popularity_score': popularity_score,
                        'category': category
                    })
            
            if not popularity_data:
                logger.warning("Aucune donnée de popularité disponible")
                return pd.DataFrame()
            
            # Créer le DataFrame
            popularity_df = pd.DataFrame(popularity_data)
            
            # Normaliser les scores entre 0 et 1
            if len(popularity_df) > 0:
                popularity_df['normalized_score'] = (
                    popularity_df['popularity_score'] - popularity_df['popularity_score'].min()
                ) / (popularity_df['popularity_score'].max() - popularity_df['popularity_score'].min())
            
            return popularity_df.sort_values('popularity_score', ascending=False)
            
        except Exception as e:
            logger.error(f"Erreur dans calculate_popularity_scores: {e}")
            return pd.DataFrame()
    
    def get_popular_attractions(self, k=10, min_reviews=2, category=None, 
                                use_cache=True, recency_weight=0.0):
        """
        Obtenir les attractions les plus populaires
        
        Args:
            k: Nombre d'attractions à retourner
            min_reviews: Nombre minimum de reviews
            category: Filtrer par catégorie (optionnel)
            use_cache: Utiliser le cache si disponible
            recency_weight: Poids pour la récence (0-1)
            
        Returns:
            Liste de (attraction_id, score, details)
        """
        # Vérifier le cache
        cache_key = f"{k}_{min_reviews}_{category}_{recency_weight}"
        if use_cache and self.popular_attractions_cache is not None:
            if cache_key in self.popular_attractions_cache:
                return self.popular_attractions_cache[cache_key]
        
        try:
            # Calculer les scores de popularité
            popularity_df = self.calculate_popularity_scores(min_reviews=min_reviews)
            
            if popularity_df.empty:
                logger.warning("Aucune attraction populaire trouvée")
                return []
            
            # Filtrer par catégorie si spécifié
            if category:
                popularity_df = popularity_df[popularity_df['category'] == category]
            
            # Trier par score
            popularity_df = popularity_df.sort_values('popularity_score', ascending=False)
            
            # Limiter au nombre demandé
            top_attractions = popularity_df.head(k)
            
            # Formater les résultats
            results = []
            for _, row in top_attractions.iterrows():
                attraction_id = int(row['attraction_id'])
                
                # Obtenir les détails
                attraction_info = self.data_loader.get_attraction_info(attraction_id)
                
                if not attraction_info.empty:
                    # Formater les détails
                    details = self._format_attraction_details(attraction_info)
                    details.update({
                        'popularity_score': float(row['popularity_score']),
                        'normalized_score': float(row.get('normalized_score', 0)),
                        'review_count': int(row['review_count']),
                        'avg_rating': float(row['avg_rating'])
                    })
                    
                    results.append((attraction_id, float(row['popularity_score']), details))
            
            # Mettre en cache
            if use_cache:
                if self.popular_attractions_cache is None:
                    self.popular_attractions_cache = {}
                self.popular_attractions_cache[cache_key] = results
            
            # Mettre à jour les compteurs
            self.recommendation_counts['popular'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur dans get_popular_attractions: {e}")
            return []
    
    def get_trending_attractions(self, k=10, time_window=30):
        """
        Obtenir les attractions tendance (basé sur la récence des reviews)
        
        Args:
            k: Nombre d'attractions à retourner
            time_window: Fenêtre temporelle en jours (pourrait être implémenté si des dates sont disponibles)
            
        Returns:
            Liste de (attraction_id, score, details)
        """
        try:
            # Note: Cette implémentation suppose que les données ont un champ de date
            # Pour l'instant, nous utilisons une approche simplifiée
            
            if self.data_loader.interactions is None:
                return []
            
            # Calculer le nombre de reviews par attraction
            review_counts = self.data_loader.interactions['attraction_id'].value_counts()
            
            # Prendre les plus récentes (si pas de date, utiliser les plus populaires)
            trending_ids = review_counts.head(k * 2).index.tolist()
            
            results = []
            for attraction_id in trending_ids[:k]:
                # Obtenir les détails
                attraction_info = self.data_loader.get_attraction_info(attraction_id)
                
                if not attraction_info.empty:
                    # Calculer un score tendance
                    review_count = review_counts[attraction_id]
                    score = min(review_count / 10, 1.0)  # Normaliser
                    
                    details = self._format_attraction_details(attraction_info)
                    details.update({
                        'trend_score': float(score),
                        'review_count': int(review_count)
                    })
                    
                    results.append((attraction_id, float(score), details))
            
            # Mettre en cache
            self.trending_attractions_cache = results
            
            # Mettre à jour les compteurs
            self.recommendation_counts['trending'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur dans get_trending_attractions: {e}")
            return []
    
    def get_category_popular_attractions(self, category, k=10):
        """
        Obtenir les attractions populaires par catégorie
        
        Args:
            category: Catégorie de l'attraction
            k: Nombre d'attractions à retourner
            
        Returns:
            Liste de (attraction_id, score, details)
        """
        # Vérifier le cache
        if category in self.category_popular_cache:
            return self.category_popular_cache[category][:k]
        
        try:
            # Récupérer toutes les attractions de cette catégorie
            if self.data_loader.attractions_df is None:
                return []
            
            category_attractions = self.data_loader.attractions_df[
                self.data_loader.attractions_df['category'] == category
            ]
            
            if category_attractions.empty:
                logger.warning(f"Aucune attraction trouvée pour la catégorie: {category}")
                return []
            
            # Calculer les scores de popularité pour cette catégorie
            results = []
            for _, row in category_attractions.iterrows():
                attraction_id = int(row['attraction_id'])
                
                # Obtenir les statistiques de cette attraction
                if self.data_loader.interactions is not None:
                    attraction_interactions = self.data_loader.interactions[
                        self.data_loader.interactions['attraction_id'] == attraction_id
                    ]
                    
                    if len(attraction_interactions) > 0:
                        review_count = len(attraction_interactions)
                        avg_rating = attraction_interactions['rating'].mean()
                        
                        # Score de popularité pour cette catégorie
                        score = avg_rating * np.log1p(review_count)
                        
                        details = self._format_attraction_details(row)
                        details.update({
                            'category_score': float(score),
                            'review_count': int(review_count),
                            'avg_rating': float(avg_rating)
                        })
                        
                        results.append((attraction_id, float(score), details))
            
            # Trier par score
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Mettre en cache
            self.category_popular_cache[category] = results
            
            # Mettre à jour les compteurs
            self.recommendation_counts[f'category_{category}'] += 1
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Erreur dans get_category_popular_attractions: {e}")
            return []
    
    def get_diverse_recommendations(self, k=10, diversity_weight=0.3):
        """
        Obtenir des recommandations diversifiées (mix de catégories)
        
        Args:
            k: Nombre total de recommandations
            diversity_weight: Poids pour la diversité (0-1)
            
        Returns:
            Liste de (attraction_id, score, details)
        """
        try:
            # Obtenir les catégories disponibles
            if self.data_loader.attractions_df is None:
                return []
            
            categories = self.data_loader.attractions_df['category'].dropna().unique()
            
            if len(categories) == 0:
                return self.get_popular_attractions(k=k)
            
            # Nombre d'attractions par catégorie (au moins 1, proportionnel au nombre total)
            attractions_per_category = max(1, k // len(categories))
            
            results = []
            for category in categories:
                # Obtenir les attractions populaires pour cette catégorie
                category_results = self.get_category_popular_attractions(category, k=attractions_per_category)
                
                # Ajuster les scores pour la diversité
                for i, (att_id, score, details) in enumerate(category_results):
                    # Réduire légèrement le score pour favoriser la diversité
                    adjusted_score = score * (1 - diversity_weight * (i / len(category_results)))
                    results.append((att_id, adjusted_score, details))
            
            # Trier par score ajusté
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Mettre à jour les compteurs
            self.recommendation_counts['diverse'] += 1
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Erreur dans get_diverse_recommendations: {e}")
            return self.get_popular_attractions(k=k)
    
    def get_visually_popular_attractions(self, k=10):
        """
        Obtenir des attractions populaires avec de bonnes images
        (Pour les utilisateurs qui apprécient le contenu visuel)
        
        Args:
            k: Nombre d'attractions à retourner
            
        Returns:
            Liste de (attraction_id, score, details)
        """
        try:
            # Obtenir les attractions populaires
            popular = self.get_popular_attractions(k=k*2, min_reviews=1)
            
            if not popular:
                return []
            
            # Filtrer celles qui ont des images
            results = []
            for att_id, score, details in popular:
                # Vérifier si l'attraction a des images
                image_paths = self.data_loader.get_attraction_image_paths(att_id)
                has_images = len(image_paths) > 0
                
                if has_images:
                    # Augmenter le score pour les attractions avec images
                    visual_score = score * 1.2  # Bonus de 20%
                    
                    # Mettre à jour les détails
                    details['has_images'] = True
                    details['image_count'] = len(image_paths)
                    details['visual_score'] = float(visual_score)
                    
                    results.append((att_id, float(visual_score), details))
                
                if len(results) >= k:
                    break
            
            # Si pas assez d'attractions avec images, ajouter d'autres attractions
            if len(results) < k:
                for att_id, score, details in popular:
                    if att_id not in [r[0] for r in results]:
                        details['has_images'] = False
                        details['image_count'] = 0
                        details['visual_score'] = float(score)
                        results.append((att_id, float(score), details))
                        
                    if len(results) >= k:
                        break
            
            # Trier par score visuel
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Mettre à jour les compteurs
            self.recommendation_counts['visual_popular'] += 1
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Erreur dans get_visually_popular_attractions: {e}")
            return self.get_popular_attractions(k=k)
    
    def solve_cold_start(self, user_id=None, k=10, strategy='hybrid', 
                        user_preferences=None):
        """
        Résoudre le problème de démarrage à froid pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur (optionnel)
            k: Nombre de recommandations
            strategy: Stratégie à utiliser:
                - 'popular': Attractions populaires
                - 'trending': Attractions tendance
                - 'diverse': Mix diversifié
                - 'visual': Attractions avec bonnes images
                - 'hybrid': Combinaison des stratégies
            user_preferences: Préférences utilisateur (optionnel)
                Format: {'categories': ['cat1', 'cat2'], 'visual_importance': 0.5}
                
        Returns:
            Liste de (attraction_id, score, details, strategy_used)
        """
        try:
            # Déterminer la stratégie en fonction des préférences
            if user_preferences:
                strategy = self._determine_strategy_from_preferences(user_preferences)
            
            results = []
            strategy_used = strategy
            
            if strategy == 'popular':
                results = self.get_popular_attractions(k=k)
                
            elif strategy == 'trending':
                results = self.get_trending_attractions(k=k)
                
            elif strategy == 'diverse':
                results = self.get_diverse_recommendations(k=k)
                
            elif strategy == 'visual':
                results = self.get_visually_popular_attractions(k=k)
                
            elif strategy == 'hybrid':
                # Combinaison de plusieurs stratégies
                popular = self.get_popular_attractions(k=k//2)
                diverse = self.get_diverse_recommendations(k=k//2)
                
                # Combiner et dédupliquer
                combined = {}
                for att_id, score, details in popular + diverse:
                    if att_id not in combined:
                        combined[att_id] = (score, details)
                    else:
                        # Prendre le meilleur score
                        old_score, old_details = combined[att_id]
                        if score > old_score:
                            combined[att_id] = (score, details)
                
                # Convertir en liste et trier
                results = [(att_id, score, details) 
                          for att_id, (score, details) in combined.items()]
                results.sort(key=lambda x: x[1], reverse=True)
                
            else:
                logger.warning(f"Stratégie inconnue: {strategy}, utilisation de 'popular'")
                results = self.get_popular_attractions(k=k)
                strategy_used = 'popular'
            
            # Ajouter des informations de stratégie
            final_results = []
            for att_id, score, details in results[:k]:
                details['cold_start_strategy'] = strategy_used
                final_results.append((att_id, score, details, strategy_used))
            
            # Journaliser
            logger.info(f"Cold start recommendations: user={user_id}, strategy={strategy_used}, count={len(final_results)}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Erreur dans solve_cold_start: {e}")
            # Fallback: attractions populaires
            popular = self.get_popular_attractions(k=k)
            return [(att_id, score, details, 'fallback') 
                   for att_id, score, details in popular[:k]]
    
    def get_personalized_cold_start(self, user_history=None, k=10):
        """
        Démarrage à froid personnalisé basé sur un historique limité
        
        Args:
            user_history: Historique utilisateur (DataFrame avec attraction_id et rating)
            k: Nombre de recommandations
            
        Returns:
            Liste de (attraction_id, score, details)
        """
        try:
            if user_history is None or len(user_history) == 0:
                return self.solve_cold_start(k=k, strategy='diverse')
            
            # Analyser les préférences de l'utilisateur
            user_preferences = self._analyze_user_preferences(user_history)
            
            # Obtenir des recommandations basées sur les préférences
            if 'preferred_categories' in user_preferences and user_preferences['preferred_categories']:
                # Utiliser les catégories préférées
                results = []
                for category in user_preferences['preferred_categories'][:3]:
                    category_results = self.get_category_popular_attractions(category, k=k//3)
                    results.extend(category_results)
                
                # Trier et limiter
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Ajouter des détails
                final_results = []
                for att_id, score, details in results[:k]:
                    details['personalized_cold_start'] = True
                    details['inferred_preferences'] = user_preferences
                    final_results.append((att_id, score, details, 'personalized'))
                
                return final_results
            else:
                # Pas de préférences claires, utiliser une stratégie hybride
                return self.solve_cold_start(k=k, strategy='hybrid')
                
        except Exception as e:
            logger.error(f"Erreur dans get_personalized_cold_start: {e}")
            return self.solve_cold_start(k=k, strategy='popular')
    
    def get_recommendation_for_new_user(self, k=10, include_explanation=True):
        """
        Obtenir des recommandations pour un nouvel utilisateur avec explication
        
        Args:
            k: Nombre de recommandations
            include_explanation: Inclure une explication pour chaque recommandation
            
        Returns:
            Dictionnaire avec recommandations et explications
        """
        try:
            # Obtenir des recommandations diversifiées
            recommendations = self.get_diverse_recommendations(k=k)
            
            results = []
            for i, (att_id, score, details) in enumerate(recommendations[:k], 1):
                recommendation = {
                    'rank': i,
                    'attraction_id': int(att_id),
                    'score': float(score),
                    'details': details
                }
                
                if include_explanation:
                    explanation = self._generate_explanation(details, score, i)
                    recommendation['explanation'] = explanation
                
                results.append(recommendation)
            
            # Statistiques
            categories = [r['details'].get('category', 'Unknown') for r in results]
            category_distribution = {cat: categories.count(cat) for cat in set(categories)}
            
            response = {
                'recommendations': results,
                'summary': {
                    'total_recommendations': len(results),
                    'strategy': 'diverse_cold_start',
                    'category_distribution': category_distribution,
                    'avg_score': np.mean([r['score'] for r in results]) if results else 0
                },
                'explanation': "Recommandations basées sur la popularité et la diversité pour les nouveaux utilisateurs."
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur dans get_recommendation_for_new_user: {e}")
            return {'error': str(e), 'recommendations': []}
    
    def _analyze_user_preferences(self, user_history):
        """
        Analyser les préférences utilisateur à partir d'un historique limité
        
        Args:
            user_history: DataFrame avec attraction_id et rating
            
        Returns:
            Dictionnaire avec préférences
        """
        preferences = {
            'preferred_categories': [],
            'avg_rating': 0,
            'rating_range': (0, 0)
        }
        
        try:
            if len(user_history) == 0:
                return preferences
            
            # Calculer la note moyenne
            preferences['avg_rating'] = float(user_history['rating'].mean())
            preferences['rating_range'] = (
                float(user_history['rating'].min()),
                float(user_history['rating'].max())
            )
            
            # Analyser les catégories
            categories = []
            for _, row in user_history.iterrows():
                attraction_info = self.data_loader.get_attraction_info(row['attraction_id'])
                if not attraction_info.empty:
                    category = attraction_info.get('category', 'Unknown')
                    if category != 'Unknown':
                        categories.append((category, row['rating']))
            
            # Trouver les catégories préférées (note moyenne > 3.5)
            category_ratings = {}
            for category, rating in categories:
                if category not in category_ratings:
                    category_ratings[category] = []
                category_ratings[category].append(rating)
            
            preferred_categories = []
            for category, ratings in category_ratings.items():
                avg_rating = np.mean(ratings)
                if avg_rating >= 3.5 and len(ratings) >= 1:
                    preferred_categories.append((category, avg_rating, len(ratings)))
            
            # Trier par note moyenne
            preferred_categories.sort(key=lambda x: x[1], reverse=True)
            preferences['preferred_categories'] = [cat for cat, _, _ in preferred_categories]
            
            return preferences
            
        except Exception as e:
            logger.error(f"Erreur dans _analyze_user_preferences: {e}")
            return preferences
    
    def _determine_strategy_from_preferences(self, preferences):
        """
        Déterminer la meilleure stratégie basée sur les préférences utilisateur
        
        Args:
            preferences: Dictionnaire de préférences
            
        Returns:
            Nom de la stratégie
        """
        if not preferences:
            return 'hybrid'
        
        # Si l'utilisateur a des catégories préférées
        if 'categories' in preferences and preferences['categories']:
            return 'diverse'
        
        # Si l'utilisateur apprécie le visuel
        if 'visual_importance' in preferences and preferences['visual_importance'] > 0.7:
            return 'visual'
        
        # Si l'utilisateur aime les tendances
        if 'prefers_trending' in preferences and preferences['prefers_trending']:
            return 'trending'
        
        # Par défaut: hybride
        return 'hybrid'
    
    def _format_attraction_details(self, attraction_info):
        """
        Formater les détails d'une attraction
        
        Args:
            attraction_info: Series ou dictionnaire avec les informations de l'attraction
            
        Returns:
            Dictionnaire formaté
        """
        try:
            details = {
                'name': self._clean_attraction_name(attraction_info.get('attraction_url', 'Unknown')),
                'category': str(attraction_info.get('category', 'Unknown')),
                'rating': float(attraction_info.get('rating', 0))
            }
            
            # Ajouter des informations supplémentaires si disponibles
            if 'latitude' in attraction_info and 'longitude' in attraction_info:
                try:
                    details['latitude'] = float(attraction_info.get('latitude', 0))
                    details['longitude'] = float(attraction_info.get('longitude', 0))
                except:
                    pass
            
            if 'address' in attraction_info:
                details['address'] = str(attraction_info.get('address', ''))
            
            if 'description' in attraction_info:
                details['description'] = str(attraction_info.get('description', ''))[:200]  # Limiter la longueur
            
            # Vérifier les images
            attraction_id = attraction_info.get('attraction_id', 0)
            if isinstance(attraction_id, (int, np.integer)):
                image_paths = self.data_loader.get_attraction_image_paths(int(attraction_id))
                details['has_images'] = len(image_paths) > 0
                details['image_count'] = len(image_paths)
            
            return details
            
        except Exception as e:
            logger.error(f"Erreur dans _format_attraction_details: {e}")
            return {
                'name': f"Attraction {attraction_info.get('attraction_id', 'Unknown')}",
                'category': 'Unknown',
                'rating': 0.0
            }
    
    def _clean_attraction_name(self, url):
        """Nettoyer le nom d'une attraction à partir de l'URL"""
        if not isinstance(url, str):
            return "Unknown Attraction"
        
        name = url
        if 'Reviews-' in name:
            name = name.split('Reviews-')[1]
        if '.html' in name:
            name = name.replace('.html', '')
        
        name = name.replace('_', ' ').replace('-', ' ').title()
        return name[:100]  # Limiter la longueur
    
    def _generate_explanation(self, details, score, rank):
        """
        Générer une explication pour une recommandation
        
        Args:
            details: Détails de l'attraction
            score: Score de recommandation
            rank: Rang de la recommandation
            
        Returns:
            Explication textuelle
        """
        explanations = []
        
        # Basé sur la popularité
        if score > 0.8:
            explanations.append("Très populaire auprès des visiteurs")
        elif score > 0.6:
            explanations.append("Bien noté par la communauté")
        
        # Basé sur la catégorie
        category = details.get('category', 'Unknown')
        if category != 'Unknown':
            explanations.append(f"Meilleure attraction dans la catégorie {category}")
        
        # Basé sur les images
        if details.get('has_images', False):
            image_count = details.get('image_count', 0)
            if image_count > 3:
                explanations.append("Excellentes photos disponibles")
            else:
                explanations.append("Photos disponibles")
        
        # Basé sur le rang
        if rank <= 3:
            explanations.append("Parmi les meilleures recommandations")
        
        # Si aucune explication spécifique, utiliser une générique
        if not explanations:
            explanations.append("Recommandation basée sur la popularité et les notes des utilisateurs")
        
        return ". ".join(explanations) + "."
    
    def get_statistics(self):
        """
        Obtenir des statistiques sur les recommandations de démarrage à froid
        
        Returns:
            Dictionnaire avec statistiques
        """
        return {
            'total_recommendations': sum(self.recommendation_counts.values()),
            'recommendation_counts': dict(self.recommendation_counts),
            'cache_status': {
                'popular_cache': len(self.popular_attractions_cache) if self.popular_attractions_cache else 0,
                'category_cache': len(self.category_popular_cache),
                'trending_cache': len(self.trending_attractions_cache) if self.trending_attractions_cache else 0
            }
        }
    
    def clear_cache(self):
        """Effacer tous les caches"""
        self.popular_attractions_cache = None
        self.category_popular_cache = {}
        self.trending_attractions_cache = None
        logger.info("Caches du ColdStartSolver effacés")
    
    def save_popularity_data(self, filepath):
        """
        Sauvegarder les données de popularité dans un fichier
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        try:
            popularity_df = self.calculate_popularity_scores()
            
            if not popularity_df.empty:
                # Sauvegarder en CSV
                popularity_df.to_csv(filepath, index=False)
                logger.info(f"Données de popularité sauvegardées dans {filepath}")
                
                # Sauvegarder un résumé en JSON
                summary = {
                    'total_attractions': len(popularity_df),
                    'top_10_attractions': popularity_df.head(10)[['attraction_id', 'popularity_score', 'category']].to_dict('records'),
                    'statistics': {
                        'avg_popularity_score': float(popularity_df['popularity_score'].mean()),
                        'max_popularity_score': float(popularity_df['popularity_score'].max()),
                        'min_popularity_score': float(popularity_df['popularity_score'].min())
                    }
                }
                
                summary_path = filepath.replace('.csv', '_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                return True
            else:
                logger.warning("Aucune donnée de popularité à sauvegarder")
                return False
                
        except Exception as e:
            logger.error(f"Erreur dans save_popularity_data: {e}")
            return False


# Exemple d'utilisation
def create_cold_start_solver(data_loader, attractions_features=None):
    """
    Créer une instance de ColdStartSolver
    
    Args:
        data_loader: Instance de TourismDataLoader
        attractions_features: Features CNN (optionnel)
        
    Returns:
        Instance de ColdStartSolver
    """
    return ColdStartSolver(data_loader, attractions_features)


if __name__ == "__main__":
    # Exemple d'utilisation
    print("=" * 60)
    print("ColdStartSolver - Démonstration")
    print("=" * 60)
    
    # Note: Ce script nécessite que le système principal soit chargé
    print("\nPour utiliser ColdStartSolver:")
    print("""
    1. Chargez d'abord votre système:
       from main import main
       system_components = main()
    
    2. Créez le solveur:
       from cold_start_solver import create_cold_start_solver
       cold_start_solver = create_cold_start_solver(
           system_components['data_loader'],
           system_components['attractions_features']
       )
    
    3. Utilisez les méthodes:
       # Attractions populaires
       popular = cold_start_solver.get_popular_attractions(k=10)
       
       # Recommandations pour nouveau utilisateur
       recommendations = cold_start_solver.solve_cold_start(k=5, strategy='hybrid')
       
       # Recommandations personnalisées
       personalized = cold_start_solver.get_personalized_cold_start(user_history, k=5)
    """)
    
    print("\nFonctionnalités disponibles:")
    print("  • get_popular_attractions() - Attractions les plus populaires")
    print("  • get_trending_attractions() - Attractions tendance")
    print("  • get_category_popular_attractions() - Populaire par catégorie")
    print("  • get_diverse_recommendations() - Recommandations diversifiées")
    print("  • get_visually_popular_attractions() - Attractions avec bonnes images")
    print("  • solve_cold_start() - Résolution complète du démarrage à froid")
    print("  • get_personalized_cold_start() - Personnalisation basée sur historique limité")
    print("  • get_recommendation_for_new_user() - Recommandations avec explications")