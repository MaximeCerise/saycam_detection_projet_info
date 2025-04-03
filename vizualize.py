"""
Module de visualisation des résultats de détection d'objets.
Ce module permet de générer une visualisation interactive des trajectoires d'objets détectés
dans une séquence vidéo, avec la possibilité de visualiser les images correspondantes.
"""

import os
import pandas as pd
import plotly.graph_objects as go

def preprocess_data(df):
    """
    Prétraite les données pour la visualisation.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données de détection
        
    Returns:
        pd.DataFrame: DataFrame prétraité avec les colonnes nécessaires
    """
    # Gestion de l'index temporel
    if 'frame' in df.columns:
        df['time_idx'] = df['frame']
    elif 'time_idx' not in df.columns:
        df['time_idx'] = df.index

    # Calcul des centres des bounding boxes
    if 'xmin' in df.columns and 'xmax' in df.columns:
        df['x_center'] = (df['xmin'] + df['xmax']) / 2
    else:
        df['x_center'] = df.index

    if 'ymin' in df.columns and 'ymax' in df.columns:
        df['y_center'] = (df['ymin'] + df['ymax']) / 2
    else:
        df['y_center'] = df.index

    # Gestion des noms d'objets
    if 'name' not in df.columns:
        df['name'] = "Objet"

    return df

def create_traces(object_names):
    """
    Crée les traces pour chaque objet à visualiser.
    
    Args:
        object_names (list): Liste des noms d'objets uniques
        
    Returns:
        list: Liste des traces Plotly
    """
    traces = []
    for obj_name in object_names:
        traces.append(
            go.Scatter(
                x=[],
                y=[],
                mode='lines+markers',
                name=obj_name,
                showlegend=True,
                hovertemplate=(
                    "<b>%{meta}</b><br>"
                    "X: %{x}<br>"
                    "Y: %{y}<br>"
                    "<extra></extra>"
                ),
                meta=obj_name,
                marker=dict(size=8)
            )
        )
    return traces

def create_frames(df, object_names, all_times, images_folder):
    """
    Crée les frames pour l'animation.
    
    Args:
        df (pd.DataFrame): DataFrame des données
        object_names (list): Liste des noms d'objets
        all_times (list): Liste des indices temporels
        images_folder (str): Chemin vers le dossier des images
        
    Returns:
        list: Liste des frames Plotly
    """
    frames = []
    for t in all_times:
        df_cumul = df[df['time_idx'] <= t]
        data_frame = []
        for obj_name in object_names:
            df_obj = df_cumul[df_cumul['name'] == obj_name]
            x_vals = df_obj['x_center'].tolist()
            y_vals = df_obj['y_center'].tolist()
            custom_vals = df_obj['image'].tolist()
            data_frame.append(
                dict(
                    x=x_vals,
                    y=y_vals,
                    customdata=custom_vals
                )
            )
        frames.append(go.Frame(name=str(t), data=data_frame))
    return frames

def create_slider_steps(all_times):
    """
    Crée les étapes du slider pour l'animation.
    
    Args:
        all_times (list): Liste des indices temporels
        
    Returns:
        list: Liste des étapes du slider
    """
    return [
        dict(
            method='animate',
            label=str(t),
            args=[
                [str(t)],
                dict(mode='immediate', frame=dict(duration=300, redraw=True), transition=dict(duration=0))
            ]
        ) for t in all_times
    ]

def generate_visualization(video_path="vid3_frames", data_path="data3.csv"):
    """
    Fonction principale qui génère la visualisation complète.
    
    Args:
        video_path (str): Chemin vers le dossier contenant les images de la vidéo
        data_path (str): Chemin vers le fichier CSV contenant les données de détection
    """
    # Lecture des données
    df = pd.read_csv(data_path)
    images_folder = video_path
    
    # Prétraitement des données
    df = preprocess_data(df)
    df['image'] = df['time_idx'].apply(lambda t: os.path.join(images_folder, f"viz_frame_{int(t)}.jpg"))
    
    # Préparation des données pour la visualisation
    object_names = sorted(df['name'].unique())
    all_times = sorted(df['time_idx'].unique())
    
    # Création des composants de la visualisation
    traces = create_traces(object_names)
    frames = create_frames(df, object_names, all_times, images_folder)
    slider_steps = create_slider_steps(all_times)
    
    # Configuration du slider
    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "Frame : "},
        "pad": {"t": 50},
        "steps": slider_steps
    }]
    
    # Configuration des boutons de contrôle
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            x=0.5,
            y=1.15,
            xanchor="center",
            yanchor="top",
            direction="left",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300},
                        },
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                ),
            ],
        )
    ]
    
    # Création de la figure
    fig = go.Figure(data=traces, frames=frames)
    
    # Mise à jour du layout
    fig.update_layout(
        title="Visualisation des résultats",
        width=1000,
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis=dict(range=[0, max(df['x_center'])+50]),
        yaxis=dict(range=[0, max(df['y_center'])+50]),
        sliders=sliders,
        updatemenus=updatemenus,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='auto')
    )
    
    # Génération du HTML
    html_fig = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        div_id='plot'
    )
    
    # Template HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Visualisation des résultats</title>
        <style>
            body {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
                margin: 0;
                padding: 0;
                background-color: #f8f8f8;
            }}
            h1 {{
                text-align: center;
                margin-top: 1em;
            }}
            #plot {{
                max-width: 1200px;
                width: 100%;
            }}
            #modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                overflow: auto;
                background-color: rgba(0,0,0,0.8);
            }}
            #modal-content {{
                background-color: #fff;
                margin: 10% auto;
                padding: 20px;
                border-radius: 4px;
                width: 80%;
                max-width: 600px;
                position: relative;
                text-align: center;
            }}
            #modal-content img {{
                max-width: 100%;
                height: auto;
            }}
            #close {{
                position: absolute;
                top: 10px;
                right: 15px;
                font-size: 28px;
                font-weight: bold;
                color: #333;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <h1>Visualisation des résultats</h1>
        {html_fig}
        <div id="modal">
            <div id="modal-content">
                <span id="close">&times;</span>
                <img id="modal-image" src="" alt="Image">
            </div>
        </div>
        <script>
            var plotDiv = document.getElementById('plot');
            plotDiv.on('plotly_click', function(data) {{
                var point = data.points[0];
                var imageSrc = point.customdata;
                if (Array.isArray(imageSrc)) {{
                    imageSrc = imageSrc[0];
                }}
                if (!imageSrc) {{
                    return;
                }}
                document.getElementById('modal-image').src = imageSrc;
                document.getElementById('modal').style.display = 'block';
            }});
            document.getElementById('close').onclick = function() {{
                document.getElementById('modal').style.display = 'none';
            }};
            window.onclick = function(event) {{
                if (event.target == document.getElementById('modal')) {{
                    document.getElementById('modal').style.display = 'none';
                }}
            }};
        </script>
    </body>
    </html>
    """
    
    # Sauvegarde du fichier HTML
    with open("simulation_visualization.html", "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    generate_visualization() 