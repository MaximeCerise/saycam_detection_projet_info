#!/usr/bin/env python3
"""
Script d'exécution pour la visualisation des résultats de détection.
Ce script permet de lancer la visualisation des données de détection d'objets.
"""

import os
import argparse
import webbrowser
from vizualize import generate_visualization

def parse_arguments():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        argparse.Namespace: Les arguments parsés
    """
    parser = argparse.ArgumentParser(description='Visualisation des résultats de détection')
    parser.add_argument('--data', '-d', type=str, default='data3.csv',
                        help='Chemin vers le fichier CSV contenant les données de détection')
    parser.add_argument('--video', '-v', type=str, default='vid3_frames',
                        help='Chemin vers le dossier contenant les images de la vidéo')
    return parser.parse_args()

def main():
    # Parse les arguments
    args = parse_arguments()
    
    # Vérification des prérequis
    if not os.path.exists(args.data):
        print(f"Erreur: Le fichier de données {args.data} n'existe pas.")
        return
    
    if not os.path.exists(args.video):
        print(f"Erreur: Le dossier des images {args.video} n'existe pas.")
        return
    
    # Génération de la visualisation
    generate_visualization(video_path=args.video, data_path=args.data)
    print("Visualisation générée avec succès !")
    
    # Ouvrir le fichier HTML dans le navigateur par défaut
    html_path = os.path.abspath("simulation_visualization.html")
    webbrowser.open('file://' + html_path)

if __name__ == "__main__":
    main() 