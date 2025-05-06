import librosa
import numpy as np

chord_templates = {
    'C':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'C#':   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'D':    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    'D#':   [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'E':    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    'F':    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    'F#':   [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    'G':    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    'G#':   [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    'A':    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'A#':   [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    'B':    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
}

y, sr = librosa.load("audio/song.mp3")
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

frames_per_second = int(librosa.time_to_frames(1.0, sr=sr))
chord_sequence = []

for i in range(0, chroma.shape[1], frames_per_second):
    chunk = chroma[:, i:i+frames_per_second]
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
    chord_sequence.append(best_chord)

for i, chord in enumerate(chord_sequence):
    print(f"{i:>2}:00s - {chord}")
