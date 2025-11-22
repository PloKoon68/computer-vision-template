import os

# Script'in bulunduğu dizini al (src/)
# Proje kök dizini (src/ bir üst dizin)


from config import load_configuration
from pipeline import Pipeline
config = load_configuration()
pipeline = Pipeline(config)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_video = os.path.join(project_root, 'data', 'input', 'videos', 'bottle_track.mp4')
output_video = os.path.join(project_root, 'data', 'output', 'bottle_track.mp4')

pipeline.process_video(input_video, output_video)