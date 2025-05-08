import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Ellipse, Circle, Rectangle, Polygon
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from matplotlib.transforms import Affine2D
import os
import time
from threading import Thread

# Set backend to TkAgg for better animation support
mpl.use('TkAgg')

# Try to import pygame, but handle the case if it's not installed
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    print("Pygame not found. Music will be disabled.")
    print("To enable music, install pygame with: pip install pygame")
    PYGAME_AVAILABLE = False

def play_background_music():
    """
    Play a happy background tune.
    This function creates a simple melody using pygame.
    """
    if not PYGAME_AVAILABLE:
        print("Music disabled - pygame not available")
        return
    
    # Initialize pygame mixer for music
    pygame.mixer.init()
    
    # Create a directory for the music if it doesn't exist
    music_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music")
    os.makedirs(music_dir, exist_ok=True)
    
    # Path to the music file
    music_file = os.path.join(music_dir, "happy_tune.mid")
    
    # Check if we need to create the music file
    if not os.path.exists(music_file):
        try:
            # Try to download a happy tune from the internet
            import urllib.request
            print("Downloading a happy tune...")
            # URL to a simple, free MIDI file (replace with a valid URL if needed)
            url = "https://www.midiworld.com/download/29"
            urllib.request.urlretrieve(url, music_file)
        except:
            print("Could not download music. Using pygame's built-in sounds instead.")
            # If download fails, we'll use pygame's sound generation
            pygame.mixer.Sound(pygame.sndarray.make_sound(
                np.sin(2 * np.pi * np.arange(44100) * 440 / 44100).astype(np.float32)
            )).play(-1)
            return
    
    try:
        # Try to play the music file
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play(-1, start=3.0)  # -1 means loop indefinitely, start at 3 seconds
    except:
        print("Could not play the music file. Using simple tones instead.")
        # If playing the file fails, create a simple melody with beeps
        def play_tones():
            # Simple happy melody using beeps
            notes = [440, 494, 523, 587, 659, 698, 784]  # A4 to G5
            durations = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4]
            
            # Skip the first few notes to start from 3 seconds in
            skip_duration = 0
            skip_index = 0
            while skip_duration < 3.0 and skip_index < len(notes):
                skip_duration += durations[skip_index]
                skip_index += 1
                
            # Start playing from the calculated position
            while True:
                for i in range(skip_index, len(notes)):
                    note = notes[i]
                    duration = durations[i]
                    
                    # Create a simple sine wave for each note
                    sample_rate = 44100
                    t = np.linspace(0, duration, int(duration * sample_rate), False)
                    tone = np.sin(2 * np.pi * note * t)
                    tone = (tone * 32767).astype(np.int16)
                    
                    # Play the tone
                    sound = pygame.sndarray.make_sound(tone)
                    sound.play()
                    time.sleep(duration)
                
                # Reset skip_index for subsequent loops
                skip_index = 0
        
        # Start the tone player in a separate thread
        tone_thread = Thread(target=play_tones)
        tone_thread.daemon = True  # Thread will exit when main program exits
        tone_thread.start()

class DancingSmiley:
    """Class to represent a dancing smiley face with animation properties"""
    def __init__(self, ax):
        self.ax = ax
        
        # Position and movement parameters
        self.x = 0.5
        self.y = 0.5
        self.size = 0.3  # Slightly smaller to make room for legs
        self.bounce_height = random.uniform(0.2, 0.4)
        self.bounce_speed = random.uniform(0.3, 0.5)
        self.wiggle_amount = random.uniform(0.1, 0.3)
        self.wiggle_speed = random.uniform(0.35, 0.55)
        self.rotation = 0
        self.rotation_speed = random.uniform(-7, 7)
        self.time = 0
        
        # Leg parameters
        self.leg_length = 0.2
        self.leg_width = 0.05
        self.leg_dance_style = random.choice(["kick", "twist", "shuffle", "disco"])
        self.leg_speed = random.uniform(0.4, 0.6)
        
        # Arm parameters
        self.arm_length = 0.25
        self.arm_width = 0.04
        self.arm_dance_style = random.choice(["wave", "clap", "raise", "robot"])
        self.arm_speed = random.uniform(0.3, 0.5)
        
        # Choose a bright, happy color
        self.color = random.choice(['gold', 'yellow', 'lightyellow', 'orange', 'lightblue', 'lightgreen'])
        
        # Choose eye and smile styles
        self.eye_style = random.choice(["normal", "star", "heart"])
        self.smile_style = random.choice(["simple", "toothy", "rainbow"])
        self.accessory = random.choice(["none", "hat", "sunglasses"])
        
        # Create initial patches
        self.patches = []
        self.create_patches()
        
    def create_patches(self):
        """Create all the patches for the smiley face"""
        # Clear existing patches
        for patch in self.patches:
            patch.remove()
        self.patches = []
        
        # Calculate current position with bounce and wiggle
        bounce = self.bounce_height * np.sin(self.time * self.bounce_speed)
        wiggle = self.wiggle_amount * np.sin(self.time * self.wiggle_speed)
        current_x = self.x + wiggle
        current_y = self.y + bounce
        
        # Draw the legs first (so they appear behind the body)
        self.draw_dancing_legs(current_x, current_y)
        
        # Draw the arms (behind the body but in front of legs)
        self.draw_dancing_arms(current_x, current_y)
        
        # Draw the face circle
        face = Circle((current_x, current_y), self.size, fill=True, color=self.color)
        self.ax.add_patch(face)
        self.patches.append(face)
        
        # Draw eyes based on style
        if self.eye_style == "normal":
            # Simple black circle eyes
            left_eye = Circle((current_x - 0.1, current_y + 0.1), 0.05, fill=True, color='black')
            right_eye = Circle((current_x + 0.1, current_y + 0.1), 0.05, fill=True, color='black')
            self.ax.add_patch(left_eye)
            self.ax.add_patch(right_eye)
            self.patches.extend([left_eye, right_eye])
        elif self.eye_style == "star":
            # Star eyes
            left_eye = Circle((current_x - 0.1, current_y + 0.1), 0.05, fill=True, color='black')
            right_eye = Circle((current_x + 0.1, current_y + 0.1), 0.05, fill=True, color='black')
            self.ax.add_patch(left_eye)
            self.ax.add_patch(right_eye)
            self.patches.extend([left_eye, right_eye])
            
            # Add stars
            for i in range(8):
                angle = i * np.pi/4
                left_line = self.ax.plot([current_x - 0.1 + 0.03*np.cos(angle), 
                                         current_x - 0.1 + 0.07*np.cos(angle)], 
                                        [current_y + 0.1 + 0.03*np.sin(angle), 
                                         current_y + 0.1 + 0.07*np.sin(angle)], 
                                        'white', linewidth=1.5)[0]
                right_line = self.ax.plot([current_x + 0.1 + 0.03*np.cos(angle), 
                                          current_x + 0.1 + 0.07*np.cos(angle)], 
                                         [current_y + 0.1 + 0.03*np.sin(angle), 
                                          current_y + 0.1 + 0.07*np.sin(angle)], 
                                         'white', linewidth=1.5)[0]
                self.patches.extend([left_line, right_line])
        else:  # heart
            # Draw simple black eyes first
            left_eye = Circle((current_x - 0.1, current_y + 0.1), 0.05, fill=True, color='black')
            right_eye = Circle((current_x + 0.1, current_y + 0.1), 0.05, fill=True, color='black')
            self.ax.add_patch(left_eye)
            self.ax.add_patch(right_eye)
            self.patches.extend([left_eye, right_eye])
            
            # Add small hearts
            for eye_x in [current_x - 0.1, current_x + 0.1]:
                # Simple heart shape
                heart_line = self.ax.plot([eye_x - 0.02, eye_x, eye_x + 0.02], 
                                         [current_y + 0.08, current_y + 0.12, current_y + 0.08], 
                                         'white', linewidth=1.5)[0]
                self.patches.append(heart_line)
        
        # Draw smile based on style
        if self.smile_style == "simple":
            # Simple curved smile
            theta = np.linspace(0, np.pi, 100)
            x = 0.2 * np.cos(theta) + current_x
            y = -0.2 * np.sin(theta) + current_y - 0.1
            smile = self.ax.plot(x, y, 'black', linewidth=3)[0]
            self.patches.append(smile)
        elif self.smile_style == "toothy":
            # Toothy smile
            # Draw the smile outline
            theta = np.linspace(0, np.pi, 100)
            x = 0.2 * np.cos(theta) + current_x
            y = -0.2 * np.sin(theta) + current_y - 0.1
            smile = self.ax.plot(x, y, 'black', linewidth=3)[0]
            self.patches.append(smile)
            
            # Add teeth
            for i in range(-2, 3):
                tooth_x = current_x + i * 0.07
                tooth = self.ax.add_patch(Rectangle((tooth_x, current_y - 0.05), 0.06, -0.05, color='white'))
                self.patches.append(tooth)
        else:  # rainbow
            # Rainbow smile
            theta = np.linspace(0, np.pi, 100)
            x = 0.25 * np.cos(theta) + current_x
            y = -0.25 * np.sin(theta) + current_y - 0.1  # Negative sin for upward curve
            
            # Draw rainbow layers
            rainbow_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
            for i, rainbow_color in enumerate(rainbow_colors):
                width = 6 - i
                y_offset = i * 0.01
                rainbow_line = self.ax.plot(x, y + y_offset, color=rainbow_color, linewidth=width)[0]
                self.patches.append(rainbow_line)
        
        # Add a random accessory
        if self.accessory == "hat":
            # Simple party hat
            hat_color = random.choice(['red', 'blue', 'green', 'purple'])
            hat = Polygon([[current_x - 0.2, current_y + 0.3], 
                          [current_x, current_y + 0.5], 
                          [current_x + 0.2, current_y + 0.3]], 
                         color=hat_color)
            self.ax.add_patch(hat)
            self.patches.append(hat)
        elif self.accessory == "sunglasses":
            # Simple sunglasses
            sunglasses = Rectangle((current_x - 0.25, current_y + 0.05), 0.5, 0.1, color='black', alpha=0.7)
            self.ax.add_patch(sunglasses)
            self.patches.append(sunglasses)
    
    def draw_dancing_legs(self, x, y):
        """Draw legs with dancing movements"""
        # Calculate leg positions based on dance style and time
        leg_time = self.time * self.leg_speed
        
        # Base positions for legs (at bottom of circle)
        left_leg_top_x = x - 0.15
        right_leg_top_x = x + 0.15
        leg_top_y = y - self.size
        
        # Different dance styles
        if self.leg_dance_style == "kick":
            # Kicking dance - one leg kicks out while the other stays in place
            left_angle = 0.3 * np.sin(leg_time) - 0.3  # Mostly down with kicks
            right_angle = 0.3 * np.sin(leg_time + np.pi) - 0.3  # Opposite phase
            
            left_leg_bottom_x = left_leg_top_x + self.leg_length * np.sin(left_angle)
            left_leg_bottom_y = leg_top_y - self.leg_length * np.cos(left_angle)
            
            right_leg_bottom_x = right_leg_top_x + self.leg_length * np.sin(right_angle)
            right_leg_bottom_y = leg_top_y - self.leg_length * np.cos(right_angle)
            
        elif self.leg_dance_style == "twist":
            # Twisting dance - legs move in circular patterns
            left_leg_bottom_x = left_leg_top_x + 0.1 * np.cos(leg_time)
            left_leg_bottom_y = leg_top_y - self.leg_length + 0.1 * np.sin(leg_time)
            
            right_leg_bottom_x = right_leg_top_x + 0.1 * np.cos(leg_time + np.pi)
            right_leg_bottom_y = leg_top_y - self.leg_length + 0.1 * np.sin(leg_time + np.pi)
            
        elif self.leg_dance_style == "shuffle":
            # Shuffling dance - legs move side to side
            left_leg_bottom_x = left_leg_top_x + 0.15 * np.sin(leg_time)
            left_leg_bottom_y = leg_top_y - self.leg_length
            
            right_leg_bottom_x = right_leg_top_x + 0.15 * np.sin(leg_time)
            right_leg_bottom_y = leg_top_y - self.leg_length
            
        else:  # disco
            # Disco dance - legs move in and out
            left_leg_bottom_x = left_leg_top_x - 0.1 * np.abs(np.sin(leg_time))
            left_leg_bottom_y = leg_top_y - self.leg_length
            
            right_leg_bottom_x = right_leg_top_x + 0.1 * np.abs(np.sin(leg_time))
            right_leg_bottom_y = leg_top_y - self.leg_length
        
        # Draw the legs as thick lines
        left_leg = self.ax.plot([left_leg_top_x, left_leg_bottom_x], 
                               [leg_top_y, left_leg_bottom_y], 
                               color='black', linewidth=self.leg_width*100, solid_capstyle='round')[0]
        
        right_leg = self.ax.plot([right_leg_top_x, right_leg_bottom_x], 
                                [leg_top_y, right_leg_bottom_y], 
                                color='black', linewidth=self.leg_width*100, solid_capstyle='round')[0]
        
        # Draw feet
        left_foot = Circle((left_leg_bottom_x, left_leg_bottom_y), 0.03, fill=True, color='black')
        right_foot = Circle((right_leg_bottom_x, right_leg_bottom_y), 0.03, fill=True, color='black')
        
        self.ax.add_patch(left_foot)
        self.ax.add_patch(right_foot)
        
        self.patches.extend([left_leg, right_leg, left_foot, right_foot])
    
    def draw_dancing_arms(self, x, y):
        """Draw arms with dancing movements"""
        # Calculate arm positions based on dance style and time
        arm_time = self.time * self.arm_speed
        
        # Base positions for arms (at sides of circle)
        left_arm_top_x = x - self.size * 0.9
        right_arm_top_x = x + self.size * 0.9
        arm_top_y = y
        
        # Different arm dance styles
        if self.arm_dance_style == "wave":
            # Waving arms - one arm waves up and down
            left_angle = 0.5 * np.sin(arm_time) + 0.3  # Wave up and down
            right_angle = 0.3 * np.sin(arm_time + np.pi) - 0.3  # Opposite phase
            
            left_arm_end_x = left_arm_top_x - self.arm_length * np.cos(left_angle)
            left_arm_end_y = arm_top_y + self.arm_length * np.sin(left_angle)
            
            right_arm_end_x = right_arm_top_x + self.arm_length * np.cos(right_angle)
            right_arm_end_y = arm_top_y + self.arm_length * np.sin(right_angle)
            
        elif self.arm_dance_style == "clap":
            # Clapping arms - arms move together and apart
            clap_factor = np.abs(np.sin(arm_time))
            
            left_arm_end_x = left_arm_top_x + self.arm_length * 0.7 * clap_factor
            left_arm_end_y = arm_top_y + self.arm_length * 0.3
            
            right_arm_end_x = right_arm_top_x - self.arm_length * 0.7 * clap_factor
            right_arm_end_y = arm_top_y + self.arm_length * 0.3
            
        elif self.arm_dance_style == "raise":
            # Raising arms - both arms raise up and down together
            raise_factor = np.sin(arm_time)
            
            left_arm_end_x = left_arm_top_x - self.arm_length * 0.3
            left_arm_end_y = arm_top_y + self.arm_length * 0.7 * raise_factor
            
            right_arm_end_x = right_arm_top_x + self.arm_length * 0.3
            right_arm_end_y = arm_top_y + self.arm_length * 0.7 * raise_factor
            
        else:  # robot
            # Robot dance - arms move in sharp, angular patterns
            robot_angle_left = int(arm_time * 2) % 4 * (np.pi/4) - np.pi/4
            robot_angle_right = int(arm_time * 2 + 2) % 4 * (np.pi/4) - np.pi/4
            
            left_arm_end_x = left_arm_top_x - self.arm_length * np.cos(robot_angle_left)
            left_arm_end_y = arm_top_y + self.arm_length * np.sin(robot_angle_left)
            
            right_arm_end_x = right_arm_top_x + self.arm_length * np.cos(robot_angle_right)
            right_arm_end_y = arm_top_y + self.arm_length * np.sin(robot_angle_right)
        
        # Draw the arms as thick lines
        left_arm = self.ax.plot([left_arm_top_x, left_arm_end_x], 
                               [arm_top_y, left_arm_end_y], 
                               color='black', linewidth=self.arm_width*100, solid_capstyle='round')[0]
        
        right_arm = self.ax.plot([right_arm_top_x, right_arm_end_x], 
                                [arm_top_y, right_arm_end_y], 
                                color='black', linewidth=self.arm_width*100, solid_capstyle='round')[0]
        
        # Draw hands
        left_hand = Circle((left_arm_end_x, left_arm_end_y), 0.025, fill=True, color='black')
        right_hand = Circle((right_arm_end_x, right_arm_end_y), 0.025, fill=True, color='black')
        
        self.ax.add_patch(left_hand)
        self.ax.add_patch(right_hand)
        
        self.patches.extend([left_arm, right_arm, left_hand, right_hand])
    
    def update(self, frame):
        """Update the smiley face for animation"""
        # Update time
        self.time += 0.1
        
        # Update rotation
        self.rotation += self.rotation_speed
        
        # Recreate all patches with updated position
        self.create_patches()
        
        return self.patches

class Confetti:
    """Class to represent a single piece of confetti with animation properties"""
    def __init__(self, ax):
        self.ax = ax
        self.x = random.uniform(0, 1)
        self.y = random.uniform(0.8, 1.2)  # Start above the visible area
        self.size = random.uniform(0.01, 0.03)
        self.color = random.choice(['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan'])
        self.shape = random.choice(['circle', 'square', 'triangle'])
        self.speed = random.uniform(0.005, 0.02)
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-10, 10)
        self.horizontal_drift = random.uniform(-0.005, 0.005)
        self.patch = self._create_patch()
        
    def _create_patch(self):
        """Create the matplotlib patch for this confetti piece"""
        if self.shape == 'circle':
            return Circle((self.x, self.y), self.size, color=self.color)
        elif self.shape == 'square':
            # Create a rectangle with a rotation transform
            rect = Rectangle((self.x-self.size/2, self.y-self.size/2), self.size, self.size, 
                           color=self.color)
            # Apply rotation transform
            transform = Affine2D().rotate_deg_around(self.x, self.y, self.rotation) + self.ax.transData
            rect.set_transform(transform)
            return rect
        else:  # triangle
            # Create triangle
            triangle = Polygon([[self.x, self.y+self.size], 
                              [self.x-self.size, self.y-self.size], 
                              [self.x+self.size, self.y-self.size]], 
                             color=self.color)
            # Apply rotation transform
            transform = Affine2D().rotate_deg_around(self.x, self.y, self.rotation) + self.ax.transData
            triangle.set_transform(transform)
            return triangle
    
    def update(self):
        """Update the position and rotation of the confetti"""
        self.y -= self.speed
        self.x += self.horizontal_drift
        self.rotation += self.rotation_speed
        
        # Remove old patch and create a new one at updated position
        self.patch.remove()
        self.patch = self._create_patch()
        self.ax.add_patch(self.patch)
        
        # If confetti goes off-screen, reset it to the top
        if self.y < -0.1:
            self.y = random.uniform(1.0, 1.2)
            self.x = random.uniform(0, 1)
            
        return self.patch

def animate_happy_face():
    """Create an animation with a happy face and falling confetti"""
    # Start playing music in the background (if pygame is available)
    bPlaying = 0
    welcome_shown = False
    frame_count = 0
    
    while True:
        if PYGAME_AVAILABLE and not bPlaying:
            play_background_music()
            bPlaying = 1  # Set to 1 to prevent restarting music
        
        # Create figure with equal aspect ratio to ensure proper centering
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, aspect='equal')
        
        # Create a dancing smiley
        dancing_smiley = DancingSmiley(ax)
        
        # Create confetti objects
        confetti_pieces = [Confetti(ax) for _ in range(50)]
        
        # Add initial confetti
        for confetti in confetti_pieces:
            ax.add_patch(confetti.patch)
        
        # Choose a random happy message
        messages = [
            "What's cooking good looking!",
            "You're awesome!",
            "Dance like nobody's watching!",
            "Shake it off!",
            "Boogie wonderland!",
            "Let's groove tonight!",
            "Dance party!"
        ]
        
        # Set title with padding to position it properly
        title_text = ax.set_title(random.choice(messages), fontsize=18, pad=20)
        
        # Create welcome message text (initially invisible)
        welcome_text = ax.text(0.5, 0.5, "WELCOME TO MY REPO!", 
                              fontsize=36, fontweight='bold', color='red',
                              ha='center', va='center', alpha=0)
        
        def update(frame):
            """Update function for animation"""
            nonlocal welcome_shown, frame_count
            patches = []
            
            # Update frame counter
            frame_count += 1
            
            # Show welcome message after 3 seconds (60 frames at 20fps)
            if frame_count == 20 and not welcome_shown:
                welcome_text.set_alpha(1)
                welcome_shown = True
            
            # Hide welcome message after it's been shown for 3 seconds
            if frame_count == 120 and welcome_shown:
                welcome_text.set_alpha(0)
            
            # Update the dancing smiley
            smiley_patches = dancing_smiley.update(frame)
            patches.extend(smiley_patches)
            
            # Update confetti
            for confetti in confetti_pieces:
                patch = confetti.update()
                patches.append(patch)
            
            # Add text to the list of artists to update
            patches.append(welcome_text)
            patches.append(title_text)
                
            return patches
        
        # Set the limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
        
        # Define a function to stop the music when the window is closed
        def on_close(event):
            # Don't stop the music between windows, only when the program exits
            pass
        
        # Connect the close event to the function
        fig.canvas.mpl_connect('close_event', on_close)
        
        # Center the plot in the figure
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.tight_layout()
        plt.show()
        
        # Reset frame count and welcome flag for next window
        frame_count = 0
        welcome_shown = False

if __name__ == "__main__":
    # Generate an animated happy face with falling confetti and music (if available)
    animate_happy_face() 