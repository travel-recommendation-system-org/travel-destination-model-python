import pandas as pd
import requests
import os
import time
import random
import json
import re
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configuration du logging optimisée
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Désactiver les logs inutiles
logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('webdriver_manager').setLevel(logging.WARNING)

class OptimizedTripAdvisorScraper:
    def __init__(self, csv_path, output_dir="tripadvisor_images_optimized", max_workers=3):
        """
        Scraper optimisé pour 5709 attractions
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        os.makedirs(output_dir, exist_ok=True)
        
        # Lecture du CSV
        try:
            self.df = pd.read_csv(csv_path, encoding="latin1", sep=",", on_bad_lines="skip")
            logger.info(f"CSV chargé: {len(self.df)} attractions")
        except Exception as e:
            logger.error(f"Erreur lecture CSV: {e}")
            self.df = pd.DataFrame()
        
        # Cache pour les drivers (réutilisation)
        self.driver_cache = threading.local()
        
        # Statistiques
        self.stats_lock = threading.Lock()
        self.stats = {
            'pages_visited': 0,
            'images_found': 0,
            'images_downloaded': 0,
            'images_skipped': 0,
            'errors': 0,
            'completed_attractions': 0,
            'successful_extractions': 0
        }
        
        # Configuration
        self.max_images_per_attraction = 5
        self.request_timeout = 15
        self.page_load_timeout = 10
        
    def get_driver(self):
        """Récupère ou crée un driver pour le thread actuel"""
        if not hasattr(self.driver_cache, 'driver'):
            try:
                options = Options()
                # Mode headless pour performance
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                
                # Éviter la détection
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option('useAutomationExtension', False)
                
                # User-Agent réaliste
                user_agents = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ]
                options.add_argument(f'user-agent={random.choice(user_agents)}')
                
                # Désactiver les images pour plus de vitesse
                prefs = {
                    "profile.managed_default_content_settings.images": 2,
                    "profile.default_content_setting_values.notifications": 2,
                }
                options.add_experimental_option("prefs", prefs)
                
                # Utiliser ChromeDriverManager avec cache
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)
                
                # Script pour éviter la détection
                driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                    "userAgent": driver.execute_script("return navigator.userAgent").replace("Headless", "")
                })
                
                self.driver_cache.driver = driver
                logger.debug(f"Driver créé pour thread {threading.current_thread().name}")
                
            except Exception as e:
                logger.error(f"Erreur création driver: {e}")
                return None
        
        return self.driver_cache.driver
    
    def extract_image_urls_optimized(self, driver):
        """
        Méthode OPTIMISÉE pour extraire les URLs d'images
        Ciblant spécifiquement TripAdvisor
        """
        try:
            image_urls = []
            
            # STRATÉGIE 1: Attendre et chercher les galleries d'images
            try:
                # Attendre les containers d'images
                wait = WebDriverWait(driver, 5)
                
                # Chercher les galleries TripAdvisor
                gallery_selectors = [
                    "div[data-test-target='photo-viewer']",
                    "div.galleryImages",
                    "div.photo-grid",
                    "div.media-viewer",
                    "div.prw_rup.prw_common_basic_image"
                ]
                
                for selector in gallery_selectors:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        for elem in elements:
                            # Extraire les images de ce container
                            imgs = elem.find_elements(By.TAG_NAME, "img")
                            for img in imgs:
                                url = self._get_best_image_url(img)
                                if url and 'tripadvisor.com' in url and 'photo-' in url:
                                    clean_url = self._clean_image_url(url)
                                    if clean_url and clean_url not in image_urls:
                                        image_urls.append(clean_url)
                    except:
                        continue
                
            except Exception as e:
                logger.debug(f"Gallery extraction: {str(e)[:50]}")
            
            # STRATÉGIE 2: Chercher les URLs dans le HTML source
            page_source = driver.page_source
            
            # Pattern optimisé pour TripAdvisor
            patterns = [
                # URLs de haute qualité
                r'https://dynamic-media-cdn\.tripadvisor\.com/media/photo-o/[^"\s]+\.(?:jpg|jpeg|png|webp)',
                r'https://media-cdn\.tripadvisor\.com/media/photo-o/[^"\s]+\.(?:jpg|jpeg|png|webp)',
                
                # URLs média
                r'https://dynamic-media-cdn\.tripadvisor\.com/media/photo-[olws]/[^"\s]+\.(?:jpg|jpeg|png|webp)',
                r'https://media-cdn\.tripadvisor\.com/media/photo-[olws]/[^"\s]+\.(?:jpg|jpeg|png|webp)',
                
                # URLs génériques TripAdvisor
                r'https://[^"\s]*?tripadvisor\.com/media/[^"\s]+\.(?:jpg|jpeg|png|webp)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, page_source)
                for url in matches:
                    clean_url = self._clean_image_url(url)
                    if clean_url and clean_url not in image_urls:
                        image_urls.append(clean_url)
            
            # STRATÉGIE 3: Chercher les images avec attributs data
            try:
                all_imgs = driver.find_elements(By.TAG_NAME, "img")
                for img in all_imgs:
                    # Vérifier les attributs data
                    for attr in ['data-src', 'data-lazyurl', 'data-bigurl']:
                        url = img.get_attribute(attr)
                        if url and 'tripadvisor.com' in url and 'photo-' in url:
                            clean_url = self._clean_image_url(url)
                            if clean_url and clean_url not in image_urls:
                                image_urls.append(clean_url)
                                
                    # Vérifier src aussi
                    src = img.get_attribute('src')
                    if src and 'tripadvisor.com' in src and 'photo-' in src:
                        clean_url = self._clean_image_url(src)
                        if clean_url and clean_url not in image_urls:
                            image_urls.append(clean_url)
                            
            except Exception as e:
                logger.debug(f"Image attr extraction: {str(e)[:50]}")
            
            # Filtrer les doublons et garder les meilleures qualités
            unique_urls = []
            seen_bases = set()
            
            for url in image_urls:
                if not url:
                    continue
                    
                # Extraire la base de l'URL (sans paramètres)
                base_url = url.split('?')[0].split('#')[0]
                
                # Si on a déjà vu cette base, on skip
                if base_url in seen_bases:
                    continue
                seen_bases.add(base_url)
                
                # Vérifier que c'est bien une URL TripAdvisor
                if 'tripadvisor.com' not in url or 'photo-' not in url:
                    continue
                
                # Vérifier l'extension
                if not url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                
                unique_urls.append(url)
            
            # Trier par qualité (photo-o > photo-l > photo-w > photo-s)
            def quality_score(url):
                url_lower = url.lower()
                if 'photo-o' in url_lower:
                    return 4
                elif 'photo-l' in url_lower:
                    return 3
                elif 'photo-w' in url_lower:
                    return 2
                elif 'photo-s' in url_lower:
                    return 1
                else:
                    return 0
            
            unique_urls = sorted(unique_urls, key=quality_score, reverse=True)
            
            return unique_urls
            
        except Exception as e:
            logger.error(f"Erreur extraction URLs: {e}")
            return []
    
    def _get_best_image_url(self, img_element):
        """Extrait la meilleure URL d'une image"""
        try:
            # Essayer data-srcset d'abord
            data_srcset = img_element.get_attribute('data-srcset')
            if data_srcset:
                # Prendre la dernière URL (souvent la plus grande)
                urls = re.findall(r'(https?://[^\s,]+)', data_srcset)
                if urls:
                    return urls[-1]
            
            # Essayer data-src
            data_src = img_element.get_attribute('data-src')
            if data_src and 'http' in data_src:
                return data_src
            
            # Essayer srcset
            srcset = img_element.get_attribute('srcset')
            if srcset:
                urls = re.findall(r'(https?://[^\s,]+)', srcset)
                if urls:
                    return urls[-1]
            
            # Essayer src
            src = img_element.get_attribute('src')
            if src and 'http' in src:
                return src
                
        except:
            pass
        return None
    
    def _clean_image_url(self, url):
        """Nettoie et améliore une URL d'image"""
        if not url or 'http' not in url:
            return None
        
        # Supprimer les paramètres
        url = url.split('?')[0].split('#')[0]
        
        # Améliorer la qualité si possible
        if 'photo-w' in url:
            url = url.replace('photo-w', 'photo-o')
        elif 'photo-t' in url:
            url = url.replace('photo-t', 'photo-o')
        elif 'photo-s' in url:
            url = url.replace('photo-s', 'photo-l')
        
        return url
    
    def create_filename(self, attraction_name, index):
        """Crée un nom de fichier selon le format demandé"""
        # Enlever les chiffres au début et les points
        clean_name = re.sub(r'^\d+\.\s*', '', attraction_name)
        
        # Convertir en slug
        slug = clean_name.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)  # Garder lettres, chiffres, espaces, tirets
        slug = re.sub(r'\s+', '-', slug)  # Remplacer espaces par tirets
        
        # Formater le numéro
        num_str = str(index).zfill(2)
        
        return f"{slug}{num_str}.jpg"
    
    def check_attraction_completed(self, attraction_name, min_images=5):
        """Vérifie si une attraction a déjà été téléchargée"""
        # Nettoyer le nom
        clean_name = ''.join(c for c in attraction_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.lstrip('0123456789. ')
        
        # Vérifier le dossier
        attraction_dir = os.path.join(self.output_dir, clean_name[:50])
        if not os.path.exists(attraction_dir):
            return False
        
        # Compter les images
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        images = [f for f in os.listdir(attraction_dir) if f.lower().endswith(image_extensions)]
        
        return len(images) >= min_images
    
    def get_next_available_index(self, attraction_name):
        """Trouve le prochain index disponible"""
        # Nettoyer le nom
        clean_name = ''.join(c for c in attraction_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.lstrip('0123456789. ')
        
        attraction_dir = os.path.join(self.output_dir, clean_name[:50])
        if not os.path.exists(attraction_dir):
            return 1
        
        # Lister les images existantes
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        images = [f for f in os.listdir(attraction_dir) if f.lower().endswith(image_extensions)]
        
        if not images:
            return 1
        
        # Extraire les numéros
        numbers = []
        pattern = re.compile(r'.*?(\d{2})\.(?:jpg|jpeg|png|webp)$')
        
        for img in images:
            match = pattern.match(img)
            if match:
                numbers.append(int(match.group(1)))
        
        if not numbers:
            return 1
        
        return max(numbers) + 1
    
    def process_attraction(self, row, idx):
        """Traite une attraction (exécutée par chaque thread)"""
        thread_name = threading.current_thread().name
        attraction_name = str(row['attraction'])
        attraction_url = row.get('attraction_url') or row.get('reviews_url')
        
        if pd.isna(attraction_url):
            logger.warning(f"[{thread_name}] URL manquante pour {attraction_name}")
            return None
        
        logger.info(f"[{thread_name}] ▶️  Traitement [{idx+1}]: {attraction_name}")
        
        # Vérifier si déjà complète
        if self.check_attraction_completed(attraction_name, self.max_images_per_attraction):
            with self.stats_lock:
                self.stats['images_skipped'] += 1
                self.stats['completed_attractions'] += 1
            logger.info(f"[{thread_name}] ⏭️  SKIP: '{attraction_name}' déjà complète")
            return {
                'attraction': attraction_name,
                'status': 'skipped',
                'images_found': 0,
                'images_downloaded': 0,
                'image_filenames': [],
                'csv_index': idx
            }
        
        driver = self.get_driver()
        if not driver:
            logger.error(f"[{thread_name}] Impossible d'obtenir le driver")
            return None
        
        try:
            # Nettoyer le nom pour le dossier
            clean_name = ''.join(c for c in attraction_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_name = clean_name.lstrip('0123456789. ')
            
            # Créer le dossier
            attraction_dir = os.path.join(self.output_dir, clean_name[:50])
            os.makedirs(attraction_dir, exist_ok=True)
            
            # Construire l'URL
            if not attraction_url.startswith('http'):
                full_url = f"https://www.tripadvisor.com{attraction_url}"
            else:
                full_url = attraction_url
            
            logger.debug(f"[{thread_name}] URL: {full_url}")
            
            # Visiter la page
            try:
                driver.set_page_load_timeout(self.page_load_timeout)
                driver.get(full_url)
                
                with self.stats_lock:
                    self.stats['pages_visited'] += 1
                
                # Attendre un peu
                time.sleep(random.uniform(2, 4))
                
                # Essayer d'aller sur la page photos directement
                self._try_go_to_photos_page(driver, full_url)
                
                # Faire défiler pour charger les images
                self._scroll_page(driver)
                
            except Exception as e:
                logger.warning(f"[{thread_name}] Erreur navigation: {str(e)[:80]}")
                # Continuer quand même, on essaye d'extraire ce qu'on peut
            
            # Extraire les URLs d'images
            image_urls = self.extract_image_urls_optimized(driver)
            
            if not image_urls:
                logger.warning(f"[{thread_name}] Aucune URL d'image trouvée pour {attraction_name}")
                with self.stats_lock:
                    self.stats['errors'] += 1
                    self.stats['completed_attractions'] += 1
                
                return {
                    'attraction': attraction_name,
                    'status': 'no_images',
                    'images_found': 0,
                    'images_downloaded': 0,
                    'image_filenames': [],
                    'csv_index': idx,
                    'error': 'Aucune image trouvée'
                }
            
            with self.stats_lock:
                self.stats['images_found'] += len(image_urls)
                self.stats['successful_extractions'] += 1
            
            # Télécharger les images
            downloaded = 0
            image_filenames = []
            
            # Trouver le prochain index
            start_index = self.get_next_available_index(attraction_name)
            existing_count = start_index - 1
            max_to_download = max(0, self.max_images_per_attraction - existing_count)
            
            if max_to_download == 0:
                logger.info(f"[{thread_name}] ✅ Déjà {existing_count} images pour {attraction_name}")
                with self.stats_lock:
                    self.stats['completed_attractions'] += 1
                
                return {
                    'attraction': attraction_name,
                    'status': 'already_complete',
                    'images_found': len(image_urls),
                    'images_downloaded': 0,
                    'image_filenames': [],
                    'csv_index': idx
                }
            
            logger.info(f"[{thread_name}] 📥 {max_to_download} images à télécharger pour {attraction_name}")
            
            for i, img_url in enumerate(image_urls[:max_to_download]):
                try:
                    # Créer le nom de fichier
                    filename = self.create_filename(attraction_name, start_index + i)
                    filepath = os.path.join(attraction_dir, filename)
                    
                    # Vérifier si existe déjà
                    if os.path.exists(filepath):
                        logger.debug(f"[{thread_name}] Existe déjà: {filename}")
                        continue
                    
                    # Télécharger
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Referer': 'https://www.tripadvisor.com/',
                        'Accept': 'image/*,*/*;q=0.8',
                        'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
                    }
                    
                    response = requests.get(img_url, headers=headers, timeout=self.request_timeout, stream=True)
                    
                    if response.status_code == 200:
                        # Sauvegarder
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        # Vérifier la taille
                        if os.path.getsize(filepath) > 10240:  # 10KB minimum
                            downloaded += 1
                            image_filenames.append(filename)
                            logger.info(f"[{thread_name}] ✓ {filename} téléchargée")
                            
                            with self.stats_lock:
                                self.stats['images_downloaded'] += 1
                        else:
                            os.remove(filepath)
                            logger.warning(f"[{thread_name}] Image trop petite: {filename}")
                    
                    else:
                        logger.warning(f"[{thread_name}] HTTP {response.status_code} pour {img_url[:50]}...")
                    
                    # Petite pause entre images
                    time.sleep(random.uniform(0.5, 1.0))
                    
                except Exception as e:
                    logger.warning(f"[{thread_name}] Erreur téléchargement image {i+1}: {str(e)[:50]}")
                    with self.stats_lock:
                        self.stats['errors'] += 1
            
            # Déterminer le statut
            if downloaded >= max_to_download:
                status = 'complete'
            elif downloaded > 0:
                status = 'partial'
            else:
                status = 'failed'
            
            with self.stats_lock:
                self.stats['completed_attractions'] += 1
            
            result = {
                'attraction': attraction_name,
                'status': status,
                'url': attraction_url,
                'images_found': len(image_urls),
                'images_downloaded': downloaded,
                'image_filenames': image_filenames,
                'csv_index': idx,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'thread': thread_name
            }
            
            # Ajouter les métadonnées du CSV
            for col in ['rate', 'reviews', 'details']:
                if col in row:
                    result[col] = row[col]
            
            logger.info(f"[{thread_name}] ✅ {attraction_name}: {downloaded}/{max_to_download} images ({status})")
            return result
            
        except Exception as e:
            logger.error(f"[{thread_name}] Erreur traitement {attraction_name}: {str(e)[:100]}")
            with self.stats_lock:
                self.stats['errors'] += 1
                self.stats['completed_attractions'] += 1
            
            return {
                'attraction': attraction_name,
                'status': 'error',
                'images_found': 0,
                'images_downloaded': 0,
                'image_filenames': [],
                'csv_index': idx,
                'error': str(e),
                'thread': thread_name
            }
    
    def _try_go_to_photos_page(self, driver, url):
        """Essaye d'aller sur la page des photos"""
        try:
            # Modifier l'URL pour aller directement aux photos
            if '/Attraction_Review-' in url:
                # Format: ...-Reviews-XXXXX.html
                # Changer en: ...-Photos-XXXXX.html
                photos_url = url.replace('-Reviews-', '-Photos-')
                
                # Vérifier si c'est une URL valide
                if photos_url != url:
                    driver.get(photos_url)
                    time.sleep(random.uniform(2, 3))
                    logger.debug("Redirection vers page photos")
                    
        except Exception as e:
            logger.debug(f"Impossible d'aller sur page photos: {str(e)[:50]}")
    
    def _scroll_page(self, driver):
        """Fait défiler la page pour charger les images"""
        try:
            # Faire défiler plusieurs fois
            for i in range(3):
                scroll_height = driver.execute_script("return document.body.scrollHeight")
                scroll_position = scroll_height * (i + 1) / 4
                driver.execute_script(f"window.scrollTo(0, {scroll_position});")
                time.sleep(random.uniform(0.5, 1.5))
        except:
            pass
    
    def scrape_all(self, max_attractions=None, start_from=0):
        """Lance le scraping de toutes les attractions"""
        logger.info(f"\n{'='*70}")
        logger.info(f"TRIPADVISOR SCRAPER OPTIMISÉ")
        logger.info(f"Threads: {self.max_workers}")
        logger.info(f"Attractions totales: {len(self.df)}")
        
        # Sélectionner les attractions
        attractions = self.df.iloc[start_from:]
        
        if max_attractions:
            attractions = attractions.head(max_attractions)
        
        logger.info(f"Attractions à traiter: {len(attractions)}")
        logger.info(f"Démarrage depuis l'index: {start_from}")
        logger.info(f"{'='*70}\n")
        
        results = []
        start_time = time.time()
        
        # Utiliser ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='TripScraper') as executor:
            # Soumettre les tâches
            future_to_idx = {}
            for idx, row in attractions.iterrows():
                future = executor.submit(self.process_attraction, row, idx)
                future_to_idx[future] = idx
            
            # Suivre la progression
            with tqdm(total=len(attractions), desc="Attractions") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result(timeout=180)  # 3 minutes timeout
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Timeout/erreur pour index {idx}: {e}")
                        results.append({
                            'csv_index': idx,
                            'status': 'timeout',
                            'error': str(e)
                        })
                    
                    pbar.update(1)
                    
                    # Afficher les stats périodiquement
                    if pbar.n % 10 == 0:
                        self._print_progress_stats(start_time, pbar.n, len(attractions))
        
        # Nettoyer les drivers
        self._cleanup_drivers()
        
        # Sauvegarder les résultats
        self._save_results(results)
        
        # Afficher les stats finales
        self._print_final_stats(start_time, results)
        
        return results
    
    def _print_progress_stats(self, start_time, current, total):
        """Affiche les statistiques de progression"""
        elapsed = time.time() - start_time
        if elapsed > 0:
            speed = current / elapsed * 3600  # attractions/heure
            eta = (total - current) / (current / elapsed) if current > 0 else 0
            
            with self.stats_lock:
                logger.info(f"Progression: {current}/{total} ({current/total*100:.1f}%)")
                logger.info(f"Vitesse: {speed:.1f} attractions/heure")
                logger.info(f"ETA: {eta/3600:.1f} heures")
                logger.info(f"Images trouvées: {self.stats['images_found']}")
                logger.info(f"Images téléchargées: {self.stats['images_downloaded']}")
                logger.info(f"Extractions réussies: {self.stats['successful_extractions']}/{current}")
                logger.info("-" * 40)
    
    def _cleanup_drivers(self):
        """Nettoie tous les drivers"""
        try:
            # Nettoyer le cache de drivers
            if hasattr(self.driver_cache, 'driver'):
                try:
                    self.driver_cache.driver.quit()
                except:
                    pass
                delattr(self.driver_cache, 'driver')
        except:
            pass
    
    def _save_results(self, results):
        """Sauvegarde les résultats"""
        if not results:
            return
        
        # Sauvegarder en CSV
        summary_data = []
        for r in results:
            summary_data.append({
                'csv_index': r.get('csv_index'),
                'attraction': r.get('attraction'),
                'status': r.get('status', 'unknown'),
                'images_found': r.get('images_found', 0),
                'images_downloaded': r.get('images_downloaded', 0),
                'image_filenames': ', '.join(r.get('image_filenames', [])),
                'error': r.get('error', ''),
                'thread': r.get('thread', '')
            })
        
        df_summary = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.output_dir, 'scraping_summary.csv')
        
        # Ajouter aux résultats existants
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            df_summary = pd.concat([existing_df, df_summary], ignore_index=True)
        
        df_summary.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Résumé sauvegardé: {csv_path}")
        
        # Sauvegarder en JSON (complet)
        json_path = os.path.join(self.output_dir, 'scraping_results.json')
        all_results = []
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except:
                all_results = []
        
        all_results.extend(results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Résultats complets: {json_path}")
    
    def _print_final_stats(self, start_time, results):
        """Affiche les statistiques finales"""
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        total_downloaded = sum(r.get('images_downloaded', 0) for r in results)
        total_found = sum(r.get('images_found', 0) for r in results)
        total_skipped = sum(1 for r in results if r.get('status') in ['skipped', 'already_complete'])
        successful = sum(1 for r in results if r.get('status') in ['complete', 'partial'])
        
        logger.info(f"\n{'='*70}")
        logger.info("STATISTIQUES FINALES")
        logger.info(f"{'='*70}")
        logger.info(f"Temps total: {hours}h {minutes}m {seconds}s")
        logger.info(f"Attractions traitées: {len(results)}")
        logger.info(f"Attractions réussies: {successful}")
        logger.info(f"Attractions skipées: {total_skipped}")
        logger.info(f"Images trouvées: {total_found}")
        logger.info(f"Images téléchargées: {total_downloaded}")
        
        if elapsed > 0:
            speed = len(results) / elapsed * 3600
            logger.info(f"Vitesse moyenne: {speed:.1f} attractions/heure")
        
        logger.info(f"{'='*70}")
        logger.info(f"Dossier de sortie: {os.path.abspath(self.output_dir)}")


def main():
    """Fonction principale"""
    print("\n" + "="*70)
    print("TRIPADVISOR SCRAPER ULTRA-OPTIMISÉ")
    print("="*70)
    
    # Configuration
    CSV_PATH = "./attractions.csv"
    
    try:
        # Lire le CSV pour savoir combien d'attractions
        df_test = pd.read_csv(CSV_PATH, encoding="latin1", sep=",", on_bad_lines="skip", nrows=1)
        total_attractions = len(pd.read_csv(CSV_PATH, encoding="latin1", sep=",", on_bad_lines="skip"))
        
        print(f"Attractions détectées: {total_attractions}")
        print("\n⚠️  RECOMMANDATIONS POUR 5709 ATTRACTIONS:")
        print("   • 3 threads: ~50-60 heures (stable)")
        print("   • 5 threads: ~30-40 heures (risque modéré)")
        print("   • 8 threads: ~20-25 heures (risque élevé)")
        
        # Configuration
        max_workers = int(input(f"\nNombre de threads (3-8, recommandé 3): ") or "3")
        max_workers = max(1, min(max_workers, 10))
        
        start_from = int(input(f"Index de départ (0 pour début): ") or "0")
        
        max_att = input(f"Nombre d'attractions (vide pour toutes): ")
        max_attractions = int(max_att) if max_att.strip() else None
        
        print(f"\nConfiguration finale:")
        print(f"  • Threads: {max_workers}")
        print(f"  • Départ: {start_from}")
        print(f"  • Limite: {'toutes' if max_attractions is None else max_attractions}")
        print("="*70)
        
        # Créer et lancer le scraper
        scraper = OptimizedTripAdvisorScraper(
            csv_path=CSV_PATH,
            output_dir="tripadvisor_images_final",
            max_workers=max_workers
        )
        
        results = scraper.scrape_all(
            max_attractions=max_attractions,
            start_from=start_from
        )
        
        # Résumé final
        if results:
            total_downloaded = sum(r.get('images_downloaded', 0) for r in results)
            if total_downloaded > 0:
                print(f"\n✅ SUCCÈS! {total_downloaded} images téléchargées")
            else:
                print(f"\n⚠️  Aucune image téléchargée")
                print("   Essayez de réduire le nombre de threads (3 max)")
                print("   ou testez manuellement une page dans Chrome")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Installation des dépendances si nécessaire
    import subprocess
    import sys
    
    required = [
        'selenium',
        'webdriver-manager',
        'pandas',
        'requests',
        'tqdm'
    ]
    
    print("Vérification des dépendances...")
    for package in required:
        try:
            if package == 'webdriver-manager':
                __import__('webdriver_manager')
            else:
                __import__(package)
        except ImportError:
            print(f"Installation de {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    main()