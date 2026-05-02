"""
Interactive Audio Player with Real-time Tracking Marker
Standalone Tkinter Application for PersonalVAD Demo
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import threading
import time


class AudioPlayerApp:
    """Interactive audio player with synchronized tracking marker across multiple plots"""
    
    def __init__(self, audio, ground_truth, predictions, probabilities, sample_rate=16000):
        """
        Initialize the audio player application
        
        Args:
            audio: Audio samples (numpy array)
            ground_truth: Ground truth labels (numpy array)
            predictions: Model predictions (numpy array)
            probabilities: Prediction probabilities (numpy array, shape: [frames, 3])
            sample_rate: Audio sample rate (default: 16000 Hz)
        """
        self.audio = audio
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.probabilities = probabilities
        self.sample_rate = sample_rate
        
        # Playback state
        self.is_playing = False
        self.current_position = 0
        self.start_time = 0
        self.update_thread = None
        self.marker_lines = []  # Store marker line objects for fast updates
        
        # Calculate time parameters
        self.audio_duration = len(audio) / sample_rate
        self.frame_duration = 0.01  # 10ms per frame
        self.time_frames = np.arange(len(ground_truth)) * self.frame_duration
        
        # Trim audio to match frame length
        max_audio_samples = int(len(ground_truth) * self.frame_duration * sample_rate)
        self.audio_trimmed = audio[:max_audio_samples]
        self.time_audio = np.linspace(0, len(self.audio_trimmed) / sample_rate, len(self.audio_trimmed))
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        """Create the tkinter GUI window"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("PersonalVAD - Interactive Audio Player")
        self.root.geometry("1200x900")
        
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create control panel frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Create buttons with styling
        style = ttk.Style()
        style.configure('Play.TButton', foreground='green')
        style.configure('Pause.TButton', foreground='orange')
        style.configure('Stop.TButton', foreground='red')
        
        self.play_btn = ttk.Button(control_frame, text="▶️ Play", command=self.on_play, style='Play.TButton')
        self.play_btn.grid(row=0, column=0, padx=5)
        
        self.pause_btn = ttk.Button(control_frame, text="⏸️ Pause", command=self.on_pause, style='Pause.TButton')
        self.pause_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop", command=self.on_stop, style='Stop.TButton')
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready to play", font=('Arial', 10))
        self.status_label.grid(row=0, column=3, padx=20)
        
        # Create matplotlib figure frame
        plot_frame = ttk.Frame(self.root)
        plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(5, 1, figsize=(12, 10))
        self.fig.suptitle('Interactive Audio Playback - Position: 0.00s / {:.2f}s'.format(self.audio_duration), 
                         fontsize=14, fontweight='bold')
        
        self.create_plots(0)
        
        # Embed matplotlib figure in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add keyboard shortcuts
        self.root.bind('<space>', lambda e: self.toggle_play_pause())
        self.root.bind('r', lambda e: self.on_stop())
        self.root.bind('<Escape>', lambda e: self.on_stop())
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_plots(self, marker_position):
        """Create all 5 plots with marker at specified position (initial setup)"""
        # Clear all axes and marker lines
        for ax in self.axes:
            ax.clear()
        self.marker_lines = []
        
        # Create custom colormap: Gray (NS=0), Yellow (NTSS=1), Green (TSS=2)
        colors = ['#808080', '#FFD700', '#00AA00']  # Gray, Gold/Yellow, Green
        cmap_labels = ListedColormap(colors)
        
        # Plot 1: Audio Waveform
        self.axes[0].plot(self.time_audio, self.audio_trimmed, linewidth=0.5, color='blue', alpha=0.7)
        line0 = self.axes[0].axvline(x=marker_position, color='red', linewidth=3, linestyle='-', alpha=0.9, label='Current Position')
        self.marker_lines.append(line0)
        self.axes[0].set_ylabel('Amplitude', fontsize=10)
        self.axes[0].set_title('Audio Waveform', fontsize=11)
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].set_xlim(0, self.audio_duration)
        self.axes[0].legend(loc='upper right', fontsize=8)
        
        # Plot 2: Ground Truth Labels (with custom colors)
        self.axes[1].imshow(self.ground_truth.reshape(1, -1), aspect='auto', cmap=cmap_labels, 
                          extent=[0, self.audio_duration, 0, 1], interpolation='nearest',
                          vmin=0, vmax=2)
        line1 = self.axes[1].axvline(x=marker_position, color='red', linewidth=3, linestyle='-', alpha=0.9)
        self.marker_lines.append(line1)
        self.axes[1].set_ylabel('Ground Truth', fontsize=10)
        self.axes[1].set_yticks([])
        self.axes[1].set_title('Ground Truth Labels (Gray=NS, Yellow=NTSS, Green=TSS)', fontsize=11)
        self.axes[1].set_xlim(0, self.audio_duration)
        
        # Plot 3: Predictions (with custom colors)
        self.axes[2].imshow(self.predictions.reshape(1, -1), aspect='auto', cmap=cmap_labels,
                          extent=[0, self.audio_duration, 0, 1], interpolation='nearest',
                          vmin=0, vmax=2)
        line2 = self.axes[2].axvline(x=marker_position, color='red', linewidth=3, linestyle='-', alpha=0.9)
        self.marker_lines.append(line2)
        self.axes[2].set_ylabel('Predictions', fontsize=10)
        self.axes[2].set_yticks([])
        self.axes[2].set_title('Model Predictions (Gray=NS, Yellow=NTSS, Green=TSS)', fontsize=11)
        self.axes[2].set_xlim(0, self.audio_duration)
        
        # Plot 4: Prediction Probabilities (using matching colors)
        prob_colors = ['#808080', '#FFD700', '#00AA00']  # Gray, Yellow, Green
        class_names = ['NS (No Speech)', 'NTSS (Non-Target)', 'TSS (Target)']
        for class_idx in range(3):
            self.axes[3].plot(self.time_frames, self.probabilities[:, class_idx], 
                        label=class_names[class_idx], alpha=0.8, linewidth=2,
                        color=prob_colors[class_idx])
        line3 = self.axes[3].axvline(x=marker_position, color='red', linewidth=3, linestyle='-', alpha=0.9)
        self.marker_lines.append(line3)
        self.axes[3].set_ylabel('Probability', fontsize=10)
        self.axes[3].set_ylim(0, 1)
        self.axes[3].set_xlim(0, self.audio_duration)
        self.axes[3].set_title('Class Probabilities', fontsize=11)
        self.axes[3].legend(loc='upper right', fontsize=9)
        self.axes[3].grid(True, alpha=0.3)
        
        # Plot 5: Match/Mismatch
        match_array = (self.ground_truth == self.predictions).astype(int)
        self.axes[4].imshow(match_array.reshape(1, -1), aspect='auto', cmap='RdYlGn',
                          extent=[0, self.audio_duration, 0, 1], interpolation='nearest', vmin=0, vmax=1)
        line4 = self.axes[4].axvline(x=marker_position, color='red', linewidth=3, linestyle='-', alpha=0.9)
        self.marker_lines.append(line4)
        self.axes[4].set_ylabel('Match', fontsize=10)
        self.axes[4].set_yticks([])
        self.axes[4].set_xlabel('Time (seconds)', fontsize=10)
        self.axes[4].set_title('Prediction Match (Red=Mismatch, Green=Match)', fontsize=11)
        self.axes[4].set_xlim(0, self.audio_duration)
        
        self.fig.tight_layout()
        
    def update_plot(self):
        """Update plots with current marker position (fast - only moves marker lines)"""
        if self.marker_lines:
            # Fast update: only move the marker lines without redrawing everything
            for line in self.marker_lines:
                line.set_xdata([self.current_position, self.current_position])
            
            # Update title
            self.fig.suptitle(f'Interactive Audio Playback - Position: {self.current_position:.2f}s / {self.audio_duration:.2f}s', 
                             fontsize=14, fontweight='bold')
            
            # Fast redraw - only the changed parts
            self.canvas.draw_idle()
            self.canvas.flush_events()
        else:
            # Fallback: full redraw (only used on first update)
            self.create_plots(self.current_position)
            self.fig.suptitle(f'Interactive Audio Playback - Position: {self.current_position:.2f}s / {self.audio_duration:.2f}s', 
                             fontsize=14, fontweight='bold')
            self.canvas.draw()
        
    def play_audio_from_position(self, start_position=0):
        """Play audio starting from a specific position"""
        start_sample = int(start_position * self.sample_rate)
        if start_sample < len(self.audio):
            sd.play(self.audio[start_sample:], self.sample_rate)
            self.start_time = time.time() - start_position
            
    def update_loop(self):
        """Background loop to update marker position in real-time (high frame rate)"""
        while self.is_playing:
            elapsed = time.time() - self.start_time
            if elapsed <= self.audio_duration:
                self.current_position = elapsed
                self.status_label.config(text=f"▶️ Playing... ({self.current_position:.2f}s / {self.audio_duration:.2f}s)")
                self.update_plot()
                time.sleep(0.008)  # ~120 FPS update rate for ultra-smooth tracking
            else:
                # Reached end of audio
                self.is_playing = False
                self.current_position = 0
                self.status_label.config(text="✅ Playback finished - Ready to play again")
                self.update_plot()
                break
                
    def on_play(self):
        """Play button handler"""
        if not self.is_playing:
            self.is_playing = True
            self.play_audio_from_position(self.current_position)
            self.status_label.config(text="▶️ Starting playback...")
            
            # Start update thread
            self.update_thread = threading.Thread(target=self.update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            
    def on_pause(self):
        """Pause button handler"""
        if self.is_playing:
            self.is_playing = False
            sd.stop()
            self.status_label.config(text=f"⏸️ Paused at {self.current_position:.2f}s")
            time.sleep(0.1)
            self.update_plot()
            
    def on_stop(self):
        """Stop button handler"""
        self.is_playing = False
        sd.stop()
        self.current_position = 0
        self.status_label.config(text="⏹️ Stopped - Ready to play")
        time.sleep(0.1)
        self.update_plot()
        
    def toggle_play_pause(self):
        """Toggle between play and pause"""
        if self.is_playing:
            self.on_pause()
        else:
            self.on_play()
            
    def on_close(self):
        """Handle window close"""
        self.is_playing = False
        sd.stop()
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Start the application"""
        print("\n✅ Audio Player Application Started!")
        print("\n🎮 Controls:")
        print("   • Click ▶️ Play button or press SPACEBAR to play/pause")
        print("   • Click ⏸️ Pause button to pause")
        print("   • Click ⏹️ Stop button or press R to stop and reset")
        print("   • Press ESC to stop")
        print("\n🔴 Watch the red line track playback position in real-time!")
        
        self.root.mainloop()


def launch_audio_player(audio, ground_truth, predictions, probabilities, sample_rate=16000):
    """
    Launch the audio player application
    
    Args:
        audio: Audio samples (numpy array)
        ground_truth: Ground truth labels (numpy array)
        predictions: Model predictions (numpy array)
        probabilities: Prediction probabilities (numpy array, shape: [frames, 3])
        sample_rate: Audio sample rate (default: 16000 Hz)
    """
    app = AudioPlayerApp(audio, ground_truth, predictions, probabilities, sample_rate)
    app.run()


if __name__ == "__main__":
    # Example usage (for testing)
    print("This module should be imported and used with launch_audio_player()")
    print("Example:")
    print("  from audio_player_app import launch_audio_player")
    print("  launch_audio_player(audio, ground_truth, predictions, probabilities)")
