"""
Script adapté pour la correspondance exacte entre noms d'attractions et dossiers d'images.
Format: "Jardin Majorelle" -> "Jardin Majorelle" (dossier existe dans images/)
"""

import pandas as pd
import os
import json
import re
from datetime import datetime
import argparse
import sys


class AttractionImageMatcher:
    def __init__(self, csv_path, images_root):
        self.csv_path = csv_path
        self.images_root = images_root
        self.df = None
        self.image_folders = []
        self.results = []
    
    def load_data(self):
        print(f"Chargement du CSV : {self.csv_path}")
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"  {len(self.df)} attractions chargees")
            return True
        except Exception as e:
            print(f"  Erreur : {e}")
            return False
    
    def list_image_folders(self):
        print(f"\nAnalyse du dossier images : {self.images_root}")
        try:
            self.image_folders = [
                f for f in os.listdir(self.images_root) 
                if os.path.isdir(os.path.join(self.images_root, f))
            ]
            print(f"  {len(self.image_folders)} dossiers d'images trouves")
            return True
        except Exception as e:
            print(f"  Erreur : {e}")
            return False
    
    def normalize_to_folder_name(self, attraction_name):
        """
        Convertit le nom d'attraction au format de nom de dossier.
        Ex: "Jardin Majorelle" -> "Jardin Majorelle" (car le dossier existe tel quel)
        
        Mais d'abord, vérifions si le dossier existe avec le nom exact.
        """
        # D'abord, chercher le nom exact
        exact_name = str(attraction_name).strip()
        
        # Ensuite, essayer différentes transformations
        transformations = [
            exact_name,  # Nom exact
            exact_name.lower(),  # Minuscules
            exact_name.upper(),  # Majuscules
            exact_name.title(),  # Title case
            # Pas besoin de remplacer espaces par tirets car vos dossiers ont des espaces
        ]
        
        return transformations
    
    def find_matching_folder(self, attraction_name):
        """
        Trouve le dossier correspondant exactement ou presque.
        """
        transformations = self.normalize_to_folder_name(attraction_name)
        
        for transformed_name in transformations:
            if transformed_name in self.image_folders:
                return transformed_name
        
        # Si aucune correspondance exacte, chercher une correspondance partielle
        cleaned_attraction = str(attraction_name).lower().strip()
        
        for folder in self.image_folders:
            cleaned_folder = str(folder).lower().strip()
            
            # Correspondance exacte après nettoyage
            if cleaned_attraction == cleaned_folder:
                return folder
            
            # Correspondance partielle (un contient l'autre)
            if cleaned_attraction in cleaned_folder or cleaned_folder in cleaned_attraction:
                # Vérifier que c'est une vraie correspondance, pas un faux positif
                words_attraction = set(cleaned_attraction.split())
                words_folder = set(cleaned_folder.split())
                
                if words_attraction and words_folder:
                    common_words = words_attraction.intersection(words_folder)
                    if len(common_words) >= 1:  # Au moins un mot en commun
                        return folder
        
        return None
    
    def get_images_from_folder(self, folder_name):
        if not folder_name:
            return []
        
        folder_path = os.path.join(self.images_root, folder_name)
        
        if not os.path.exists(folder_path):
            return []
        
        images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        try:
            for file in os.listdir(folder_path):
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in image_extensions):
                    images.append(file)
            
            images.sort()
        except Exception as e:
            print(f"  Erreur lors de la lecture du dossier {folder_name}: {e}")
        
        return images
    
    def process_attractions(self):
        print(f"\nRecherche des correspondances...")
        print(f"  Dossiers disponibles: {len(self.image_folders)}")
        print(f"  Attractions a traiter: {len(self.df)}")
        
        self.results = []
        exact_matches = 0
        partial_matches = 0
        no_matches = 0
        
        for idx, row in self.df.iterrows():
            attraction_name = row['attraction_name']
            folder_match = self.find_matching_folder(attraction_name)
            
            if folder_match:
                images = self.get_images_from_folder(folder_match)
                
                # Déterminer le type de correspondance
                cleaned_attraction = str(attraction_name).lower().strip()
                cleaned_folder = str(folder_match).lower().strip()
                
                if cleaned_attraction == cleaned_folder:
                    match_type = 'EXACT'
                    exact_matches += 1
                else:
                    match_type = 'PARTIAL'
                    partial_matches += 1
                
                result = {
                    'id': row['id'] if 'id' in row else idx + 1,
                    'attraction_name': attraction_name,
                    'matched_folder': folder_match,
                    'match_type': match_type,
                    'images_count': len(images),
                    'images_list': images,
                    'images_json': json.dumps(images) if images else "[]",
                    'status': 'MATCHED'
                }
            else:
                result = {
                    'id': row['id'] if 'id' in row else idx + 1,
                    'attraction_name': attraction_name,
                    'matched_folder': None,
                    'match_type': None,
                    'images_count': 0,
                    'images_list': [],
                    'images_json': "[]",
                    'status': 'NO_MATCH'
                }
                no_matches += 1
            
            self.results.append(result)
            
            # Afficher progression
            if (idx + 1) % 100 == 0 or (idx + 1) == len(self.df):
                print(f"  Traite {idx + 1}/{len(self.df)} attractions...", end='\r')
        
        print(f"\n  Resultats:")
        print(f"    - Correspondances exactes: {exact_matches}")
        print(f"    - Correspondances partielles: {partial_matches}")
        print(f"    - Sans correspondance: {no_matches}")
        print(f"    - Total matches: {exact_matches + partial_matches}")
    
    def update_dataframe(self):
        results_df = pd.DataFrame(self.results)
        
        # Mettre à jour seulement les colonnes nécessaires
        for col in ['images_count', 'images_list', 'images_json', 'status']:
            self.df[col] = results_df[col]
        
        # Mettre à jour images_path avec la liste JSON des images
        self.df['images_path'] = results_df['images_json']
        
        # Supprimer les colonnes inutiles
        columns_to_remove = ['matched_folder', 'match_type', 'images_json']
        for col in columns_to_remove:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col], errors='ignore')
    
    def generate_report(self):
        matched = [r for r in self.results if r['status'] == 'MATCHED']
        exact = [r for r in matched if r['match_type'] == 'EXACT']
        partial = [r for r in matched if r['match_type'] == 'PARTIAL']
        no_match = [r for r in self.results if r['status'] == 'NO_MATCH']
        
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_attractions': len(self.results),
            'exact_matches': len(exact),
            'partial_matches': len(partial),
            'total_matches': len(matched),
            'no_matches': len(no_match),
            'match_percentage': round((len(matched) / len(self.results)) * 100, 2),
            'total_images_found': sum(r['images_count'] for r in self.results),
            'exact_matches_list': [
                {
                    'attraction': r['attraction_name'],
                    'folder': r['matched_folder'],
                    'images_count': r['images_count'],
                    'images': r['images_list']
                }
                for r in exact
            ],
            'partial_matches_list': [
                {
                    'attraction': r['attraction_name'],
                    'folder': r['matched_folder'],
                    'images_count': r['images_count'],
                    'images': r['images_list']
                }
                for r in partial
            ],
            'no_matches_list': [
                r['attraction_name'] for r in no_match
            ]
        }
        
        return report
    
    def save_results(self, output_csv=None, report_file=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_csv is None:
            base_name = os.path.splitext(self.csv_path)[0]
            output_csv = f"{base_name}_final.csv"
        
        try:
            # Réorganiser les colonnes pour un meilleur ordre
            columns_order = [
                'id', 'attraction_name', 'attraction_url', 'rating', 'review_count', 
                'rating_details', 'category', 'description', 'city', 'languages_count',
                'images_folder', 'images_path', 'images_count', 'images_list', 'status'
            ]
            
            # Garder seulement les colonnes présentes dans le DataFrame
            columns_order = [col for col in columns_order if col in self.df.columns]
            
            # Ajouter les autres colonnes non listées
            other_columns = [col for col in self.df.columns if col not in columns_order]
            final_columns = columns_order + other_columns
            
            self.df = self.df[final_columns]
            
            self.df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\nCSV sauvegarde : {output_csv}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du CSV : {e}")
            output_csv = None
        
        if report_file is None:
            report_file = f"marrakech_attractions_report.txt"
        
        try:
            report = self.generate_report()
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("RAPPORT DE CORRESPONDANCE IMAGES/ATTRACTIONS (FIXED)\n")
                f.write("=" * 70 + "\n\n")
                
                f.write(f"Date d'execution : {report['timestamp']}\n")
                f.write(f"Fichier source : {self.csv_path}\n")
                f.write(f"Dossier images : {self.images_root}\n")
                f.write(f"Dossiers disponibles : {len(self.image_folders)}\n\n")
                
                f.write("STATISTIQUES :\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total d'attractions : {report['total_attractions']}\n")
                f.write(f"Correspondances exactes : {report['exact_matches']}\n")
                f.write(f"Correspondances partielles : {report['partial_matches']}\n")
                f.write(f"Total correspondances : {report['total_matches']}\n")
                f.write(f"Sans correspondance : {report['no_matches']}\n")
                f.write(f"Taux de reussite : {report['match_percentage']}%\n")
                f.write(f"Total d'images trouvees : {report['total_images_found']}\n\n")
                
                if report['exact_matches_list']:
                    f.write("CORRESPONDANCES EXACTES :\n")
                    f.write("-" * 40 + "\n")
                    for match in report['exact_matches_list'][:20]:  # Limiter à 20
                        f.write(f"\n{match['attraction']}\n")
                        f.write(f"  Dossier : {match['folder']}\n")
                        f.write(f"  Images : {match['images_count']} fichier(s)\n")
                        if match['images']:
                            f.write(f"  Fichiers : {', '.join(match['images'][:3])}")
                            if len(match['images']) > 3:
                                f.write(f"... (+{len(match['images'])-3} plus)")
                            f.write("\n")
                
                if report['partial_matches_list']:
                    f.write(f"\nCORRESPONDANCES PARTIELLES ({len(report['partial_matches_list'])}):\n")
                    f.write("-" * 40 + "\n")
                    for match in report['partial_matches_list'][:20]:  # Limiter à 20
                        f.write(f"\n{match['attraction']}\n")
                        f.write(f"  -> Dossier : {match['folder']}\n")
                        f.write(f"  Images : {match['images_count']} fichier(s)\n")
                
                if report['no_matches_list']:
                    f.write(f"\nSANS CORRESPONDANCE ({len(report['no_matches_list'])}):\n")
                    f.write("-" * 40 + "\n")
                    # Grouper par lots de 10 pour la lisibilité
                    for i in range(0, len(report['no_matches_list']), 10):
                        batch = report['no_matches_list'][i:i+10]
                        f.write("  " + ", ".join(batch) + "\n")
                
                f.write("\n" + "=" * 70 + "\n")
                f.write("FIN DU RAPPORT\n")
                f.write("=" * 70 + "\n")
            
            print(f"Rapport genere : {report_file}")
            
        except Exception as e:
            print(f"Erreur lors de la generation du rapport : {e}")
            report_file = None
        
        return output_csv, report_file
    
    def display_summary(self):
        matched = [r for r in self.results if r['status'] == 'MATCHED']
        exact = [r for r in matched if r['match_type'] == 'EXACT']
        partial = [r for r in matched if r['match_type'] == 'PARTIAL']
        
        print("\n" + "="*70)
        print("RESUME DES RESULTATS (FIXED)")
        print("="*70)
        
        print(f"\nStatistiques :")
        print(f"  - Total d'attractions : {len(self.results)}")
        print(f"  - Dossiers d'images disponibles : {len(self.image_folders)}")
        print(f"  - Correspondances exactes : {len(exact)}")
        print(f"  - Correspondances partielles : {len(partial)}")
        print(f"  - Total correspondances : {len(matched)}")
        print(f"  - Sans correspondance : {len(self.results) - len(matched)}")
        print(f"  - Taux de reussite : {round(len(matched)/len(self.results)*100, 1)}%")
        
        total_images = sum(r['images_count'] for r in self.results)
        print(f"  - Total d'images associees : {total_images}")
        
        print(f"\nColonnes finales :")
        print(f"  - Nombre de colonnes : {len(self.df.columns)}")
        print(f"  - Colonnes : {', '.join(self.df.columns.tolist())}")
        
        if exact:
            print(f"\nTop 5 des correspondances exactes :")
            for i, match in enumerate(exact[:5], 1):
                print(f"  {i}. {match['attraction_name']}")
                print(f"     Images : {match['images_count']} fichier(s)")
                if match['images_list']:
                    print(f"     Ex: {match['images_list'][0]}")
        
        print("\n" + "="*70)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Associe les images aux attractions touristiques (version fixee)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s
  %(prog)s --csv data.csv --images ./pictures
        """
    )
    
    parser.add_argument(
        '--csv',
        default='marrakech_attractions_clean.csv',
        help='Chemin vers le fichier CSV des attractions'
    )
    
    parser.add_argument(
        '--images',
        default='attractions_images',
        help='Chemin vers le dossier des images'
    )
    
    parser.add_argument(
        '--output',
        help='Chemin personnalise pour le CSV de sortie'
    )
    
    return parser.parse_args()


def main():
    print("\n" + "="*70)
    print("PROCESS_ATTRACTION_IMAGES - VERSION CORRIGEE")
    print("="*70)
    
    args = parse_arguments()
    
    if not os.path.exists(args.csv):
        print(f"\nERREUR : Le fichier CSV '{args.csv}' n'existe pas!")
        sys.exit(1)
    
    if not os.path.exists(args.images):
        print(f"\nERREUR : Le dossier images '{args.images}' n'existe pas!")
        sys.exit(1)
    
    matcher = AttractionImageMatcher(args.csv, args.images)
    
    if not matcher.load_data():
        print("\nImpossible de charger les donnees CSV.")
        sys.exit(1)
    
    if not matcher.list_image_folders():
        print("\nImpossible de lister les dossiers d'images.")
        sys.exit(1)
    
    matcher.process_attractions()
    matcher.update_dataframe()
    matcher.display_summary()
    
    output_csv, report_file = matcher.save_results(args.output, None)
    
    print("\n" + "="*70)
    print("TRAITEMENT TERMINE!")
    print("="*70)
    
    if output_csv:
        print(f"\nFichier CSV : {os.path.abspath(output_csv)}")
        print(f"Colonnes dans le CSV :")
        for i, col in enumerate(matcher.df.columns, 1):
            print(f"  {i:2}. {col}")
    
    if report_file:
        print(f"\nRapport : {os.path.abspath(report_file)}")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTraitement interrompu.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)