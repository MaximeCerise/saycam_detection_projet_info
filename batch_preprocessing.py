import os
import argparse
import subprocess
from pathlib import Path

def extract_frames_for_video(video_path, output_dir, fps=30):
    """
    Extrait les frames d'une vidéo.
    
    Args:
        video_path (str): Chemin de la vidéo
        output_dir (str): Dossier de sortie pour les frames
        fps (int): Nombre de frames par seconde à extraire
    """
    try:
        subprocess.run([
            'python', 'extract_frames.py',
            video_path,
            output_dir,
            '--fps', str(fps)
        ], check=True)
        print(f"Extraction des frames terminée pour {video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'extraction des frames de {video_path}: {e}")
        raise

def process_directory(input_base_dir, output_base_dir, fps=30):
    """
    Traite tous les fichiers vidéo dans le répertoire d'entrée.
    
    Args:
        input_base_dir (str): Répertoire contenant les vidéos à traiter
        output_base_dir (str): Répertoire parent de sortie
        fps (int): Nombre de frames par seconde à extraire
    """
    # Convertir en objets Path
    input_base_path = Path(input_base_dir)
    output_base_path = Path(output_base_dir)
    
    # Créer les répertoires de sortie
    extracted_frames_dir = output_base_path / "extracted_frames"
    preprocessed_frames_dir = output_base_path / "preprocessed_frames"
    
    for dir_path in [extracted_frames_dir, preprocessed_frames_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Compteur pour suivre le nombre de fichiers traités
    processed_files_count = 0
    
    # Parcourir tous les fichiers du répertoire d'entrée
    for input_file in input_base_path.iterdir():
        # Vérifier que c'est bien un fichier vidéo
        if input_file.is_file() and input_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            # Créer les chemins de sortie correspondants
            video_name = input_file.stem
            extracted_subdir = extracted_frames_dir / video_name
            preprocessed_subdir = preprocessed_frames_dir / video_name
            
            try:
                # Créer les sous-dossiers de sortie
                extracted_subdir.mkdir(parents=True, exist_ok=True)
                preprocessed_subdir.mkdir(parents=True, exist_ok=True)
                
                # 1. Extraction des frames
                print(f"\nExtraction des frames de : {input_file.name}")
                extract_frames_for_video(str(input_file), str(extracted_subdir), fps)
                
                # 2. Prétraitement des frames
                print(f"\nPrétraitement des frames de : {input_file.name}")
                result = subprocess.run([
                    'python', 'preprocessing.py', 
                    str(extracted_subdir),  # Dossier d'entrée (frames extraites)
                    str(preprocessed_subdir)  # Dossier de sortie (frames prétraitées)
                ], capture_output=True, text=True)
                
                # Vérifier le résultat de l'exécution
                if result.returncode == 0:
                    print(f"Fichier {input_file.name} traité avec succès ✓")
                    processed_files_count += 1
                else:
                    print(f"Erreur lors du traitement de {input_file.name}")
                    print("Sortie d'erreur :", result.stderr)
            
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {input_file.name}: {e}")
    
    # Résumé final
    print(f"\n--- Résumé ---")
    print(f"Total de fichiers traités : {processed_files_count}")

def main():
    # Configuration de l'argument parser
    parser = argparse.ArgumentParser(description='Preprocessing par lots de vidéos')
    parser.add_argument('input_dir', type=str, help='Répertoire contenant les vidéos à traiter')
    parser.add_argument('output_dir', type=str, help='Répertoire de sortie principal')
    parser.add_argument('--fps', type=int, default=30, help='Nombre de frames par seconde à extraire')
    
    args = parser.parse_args()
    
    # Lancer le traitement par lots
    process_directory(args.input_dir, args.output_dir, args.fps)

if __name__ == '__main__':
    main()