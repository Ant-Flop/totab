import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Все мажорные и минорные аккорды
def generate_chord_templates():
    major = [0, 4, 7]
    minor = [0, 3, 7]
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    templates = {}
    for i, root in enumerate(notes):
        for name, intervals in [('maj', major), ('min', minor)]:
            template = [0] * 12
            for interval in intervals:
                template[(i + interval) % 12] = 1
            templates[f"{root}{'' if name == 'maj' else 'm'}"] = template
    return templates

chord_templates = generate_chord_templates()

# Загрузка аудио
y, sr = librosa.load("audio/song2.mp3", mono=True)
duration = librosa.get_duration(y=y, sr=sr)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# Сегментация
segment_duration = 1.0
frames_per_segment = int(librosa.time_to_frames(segment_duration, sr=sr))
chord_segments = []

for i in range(0, chroma.shape[1], frames_per_segment):
    chunk = chroma[:, i:i + frames_per_segment]
    if chunk.shape[1] == 0:
        continue
    avg_chroma = np.mean(chunk, axis=1)
    best_score = -1
    best_chord = 'N/A'
    for chord, template in chord_templates.items():
        score = np.dot(avg_chroma, template)
        if score > best_score:
            best_score = score
            best_chord = chord

    start_time = librosa.frames_to_time(i, sr=sr)
    end_time = librosa.frames_to_time(i + frames_per_segment, sr=sr)
    if len(chord_segments) == 0 or chord_segments[-1][2] != best_chord:
        chord_segments.append((start_time, end_time, best_chord))
    else:
        chord_segments[-1] = (chord_segments[-1][0], end_time, best_chord)

# Визуализация
fig, ax = plt.subplots(figsize=(14, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax)
ax.set_title("Распознанные аккорды по времени")
ax.set_xlabel("Время (сек)")
ax.set_ylabel("Амплитуда")

unique_chords = sorted(set([chord for _, _, chord in chord_segments]))
chord_colors = {chord: plt.cm.tab20(i / len(unique_chords)) for i, chord in enumerate(unique_chords)}

for start, end, chord in chord_segments:
    rect = Rectangle((start, ax.get_ylim()[0]), end - start, ax.get_ylim()[1] - ax.get_ylim()[0],
                     color=chord_colors[chord], alpha=0.3)
    ax.add_patch(rect)
    ax.text((start + end) / 2, ax.get_ylim()[1]*0.9, chord, ha='center', va='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.show()
