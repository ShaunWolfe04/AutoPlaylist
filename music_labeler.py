import os
import csv
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame
from mutagen.easyid3 import EasyID3
from mutagen.oggvorbis import OggVorbis

class AudioLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Labeler")
        self.root.geometry("400x350")
        
        # Initialize Pygame Mixer
        pygame.mixer.init()
        
        # State variables
        self.playlist = []
        self.current_song_path = ""
        self.output_directory = ""
        self.labeled_count = 0
        self.target_count = 0
        self.is_playing = False
        
        self.csv_filename = "playlist_labels.csv"
        self.setup_ui()

    def setup_ui(self):
        # Frame 1: Setup (Directory & Target Count)
        self.setup_frame = tk.Frame(self.root)
        self.setup_frame.pack(pady=20)
        
        tk.Label(self.setup_frame, text="Target number of songs to label:").pack()
        self.target_entry = tk.Entry(self.setup_frame, width=10)
        self.target_entry.insert(0, "50")
        self.target_entry.pack(pady=5)
        
        tk.Button(self.setup_frame, text="Select Directories & Start", command=self.start_session).pack(pady=10)
        
        # Frame 2: Active Player (Minimal UI)
        self.player_frame = tk.Frame(self.root)
        self.track_label = tk.Label(self.player_frame, text="Now Playing: ...", wraplength=350)
        self.track_label.pack(pady=20)
        
        self.progress_label = tk.Label(self.player_frame, text="Labeled: 0 / ?")
        self.progress_label.pack(pady=5)

        # Frame 3: Labeling Interface (Hidden until song ends)
        self.label_frame = tk.Frame(self.root)
        
        self.study_var = tk.IntVar(value=0)
        self.drive_var = tk.IntVar(value=0)
        self.workout_var = tk.IntVar(value=0)
        
        self.create_slider(self.label_frame, "Study (0-3):", self.study_var).pack(pady=5)
        self.create_slider(self.label_frame, "Drive (0-3):", self.drive_var).pack(pady=5)
        self.create_slider(self.label_frame, "Workout (0-3):", self.workout_var).pack(pady=5)
        
        btn_frame = tk.Frame(self.label_frame)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Replay", command=self.play_audio).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Save & Next", command=self.save_and_next).pack(side=tk.LEFT, padx=10)

    def create_slider(self, parent, text, variable):
        frame = tk.Frame(parent)
        tk.Label(frame, text=text, width=15, anchor="w").pack(side=tk.LEFT)
        tk.Scale(frame, variable=variable, from_=0, to=3, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT)
        return frame

    def start_session(self):
        try:
            self.target_count = int(self.target_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")
            return

        messagebox.showinfo("Step 1", "First, select the INPUT directory containing your 2000 songs.")
        input_directory = filedialog.askdirectory(title="Select Input Directory")
        if not input_directory:
            return

        messagebox.showinfo("Step 2", "Next, select the OUTPUT directory where labeled songs should be copied.")
        self.output_directory = filedialog.askdirectory(title="Select Output Directory")
        if not self.output_directory:
            return

        # Read existing completed files to prevent duplicates
        completed_files = set()
        if os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    if row:
                        completed_files.add(row[0]) # filename is the first column

        # Scan directory and subdirectories
        for root_dir, _, files in os.walk(input_directory):
            for file in files:
                if file.lower().endswith(('.mp3', '.ogg')):
                    if file not in completed_files:
                        self.playlist.append(os.path.join(root_dir, file))

        if not self.playlist:
            messagebox.showinfo("Done", "No new audio files found in this directory.")
            return

        self.setup_frame.pack_forget()
        self.player_frame.pack(pady=20)
        self.update_progress()
        self.load_next_song()

    def extract_metadata(self, filepath):
        filename = os.path.basename(filepath)
        song_name, artist = "Unknown", "Unknown"
        
        try:
            if filepath.lower().endswith('.mp3'):
                audio = EasyID3(filepath)
                song_name = audio.get('title', [''])[0]
                artist = audio.get('artist', [''])[0]
            elif filepath.lower().endswith('.ogg'):
                audio = OggVorbis(filepath)
                song_name = audio.get('title', [''])[0]
                artist = audio.get('artist', [''])[0]
        except Exception:
            pass 
            
        return filename, song_name, artist

    def load_next_song(self):
        if self.labeled_count >= self.target_count or not self.playlist:
            self.finish_session()
            return

        self.current_song_path = self.playlist.pop(0)
        filename, song_name, artist = self.extract_metadata(self.current_song_path)
        
        self.track_label.config(text=f"Now Playing: {song_name}\nBy: {artist}\n({filename})")
        
        # Reset sliders
        self.study_var.set(0)
        self.drive_var.set(0)
        self.workout_var.set(0)
        
        self.label_frame.pack_forget()
        self.play_audio()

    def play_audio(self):
        pygame.mixer.music.load(self.current_song_path)
        pygame.mixer.music.play()
        self.is_playing = True
        self.check_playback_status()

    def check_playback_status(self):
        if self.is_playing:
            if not pygame.mixer.music.get_busy():
                self.is_playing = False
                self.trigger_labeling_ui()
            else:
                self.root.after(500, self.check_playback_status)

    def trigger_labeling_ui(self):
        self.root.attributes('-topmost', True)
        self.label_frame.pack(pady=10)
        self.root.attributes('-topmost', False) 

    def save_and_next(self):
        filename, song_name, artist = self.extract_metadata(self.current_song_path)
        
        # Clean spaces from metadata
        song_name = song_name.replace(" ", "_").replace(",", "") if song_name else "Blank"
        artist = artist.replace(" ", "_").replace(",", "") if artist else "Blank"
        filename = filename.replace(" ", "_").replace(",", "")
        
        row_data = f"{filename},{song_name},{artist},{self.study_var.get()},{self.drive_var.get()},{self.workout_var.get()}\n"
        
        # 1. Write to CSV
        with open(self.csv_filename, 'a', encoding='utf-8') as f:
            f.write(row_data)
            
        # 2. Copy the audio file to the output directory
        destination_path = os.path.join(self.output_directory, filename)
        if not os.path.exists(destination_path):
            shutil.copy2(self.current_song_path, destination_path)
            
        self.labeled_count += 1
        self.update_progress()
        self.load_next_song()

    def update_progress(self):
        self.progress_label.config(text=f"Labeled: {self.labeled_count} / {self.target_count}")

    def finish_session(self):
        pygame.mixer.music.stop()
        self.player_frame.pack_forget()
        self.label_frame.pack_forget()
        tk.Label(self.root, text="Session Complete!").pack(pady=50)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioLabelerApp(root)
    root.mainloop()