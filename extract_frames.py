"""
Script to extract frames from a video
Author: Maxime
Date: 2023-10-10

Description:
This script extracts frames from a video at regular intervals and saves them to an output directory.
"""

# Standard modules
import os
import argparse
from pathlib import Path

# Third-party modules
import cv2


def create_output_dir(output_dir):
    """
    Creates the output directory if it doesn't exist.

    Args:
        output_dir (str): Path to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")


def extract_frames(video_path, output_dir, frame_interval):
    """
    Extracts frames from a video and saves them to a directory.

    Args:
        video_path (str): Path to the video.
        output_dir (str): Output directory for frames.
        frame_interval (int): Interval between extracted frames.
    """
    # Create the output directory
    create_output_dir(output_dir)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}.")
        return

    frame_count = 0
    saved_frame_count = 0

    while True:
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        frame_count += 1

        # Save every n-th frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count + 1}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            print(f"Frame {saved_frame_count} saved: {frame_filename}")

    # Release the video capture object
    cap.release()
    print(f"Extraction complete. {saved_frame_count} frames saved in {output_dir}.")


def main():
    """
    Main function of the script.
    """
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Extrait les frames d\'une vidéo')
    parser.add_argument('video_path', type=str, help='Chemin de la vidéo à traiter')
    parser.add_argument('output_dir', type=str, help='Dossier de sortie pour les frames')
    parser.add_argument('--fps', type=int, default=30, help='Nombre de frames par seconde à extraire')
    
    args = parser.parse_args()
    
    # Calcul de l'intervalle entre les frames
    frame_interval = int(30 / args.fps)  # 30 est le FPS standard des vidéos
    
    # Extraction des frames
    extract_frames(args.video_path, args.output_dir, frame_interval)


if __name__ == "__main__":
    main()