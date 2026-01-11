import os
import shutil

# 📂 Chemin du dossier principal
base_dir = "tripadvisor_images_final"   # change le chemin si nécessaire

# 🖼️ Extensions d’images acceptées
image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# 📊 Compteurs
deleted = 0
kept = 0

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    if os.path.isdir(folder_path):
        # Liste des fichiers image dans le dossier
        images = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(image_extensions)
        ]

        if len(images) == 0:
            # ❌ Supprimer le dossier vide (sans images)
            shutil.rmtree(folder_path)
            deleted += 1
            print(f"🗑️ Supprimé : {folder_name}")
        else:
            kept += 1
            print(f"✅ Gardé : {folder_name} ({len(images)} images)")

print("\n📌 Résumé :")
print(f"✔️ Dossiers conservés : {kept}")
print(f"🗑️ Dossiers supprimés : {deleted}")