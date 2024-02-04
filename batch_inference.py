import os
import argparse
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
from moviepy.audio.fx.all import audio_normalize
from moviepy.audio.AudioClip import AudioClip

def find_matching_files(folder_path):
    """
    Find matching video and audio files in the specified folder.

    Parameters:
    - folder_path: str, path to the folder containing video and audio files.

    Returns:
    - A list of tuples, each containing the paths to a matching video and audio file.
    """
    videos = {}  # Dictionary to hold video files with basename as key
    audios = {}  # Dictionary to hold audio files with basename as key
    matches = []  # List to hold tuples of matched video and audio file paths

    # Scan the folder for video and audio files
    for file in os.listdir(folder_path):
        # Construct absolute path
        file_path = os.path.join(folder_path, file)
        # Split the filename and extension
        basename, extension = os.path.splitext(file)
        # Check for video files (MP4)
        if extension.lower() == '.mp4':
            videos[basename] = file_path
        # Check for audio files (MP3)
        elif extension.lower() == '.mp3':
            audios[basename] = file_path

    # Match video and audio files based on basename
    for basename, video_path in videos.items():
        audio_path = audios.get(basename)
        if audio_path:  # If a matching audio file is found
            # If video_path-output exists, skip
            if os.path.exists(os.path.splitext(video_path)[0] + "-output.mp4"):
                print(f"Skipping {video_path} as output exists")
                continue
            matches.append((video_path, audio_path))

    return matches

def generate_silence(duration, fps, n_channels=2):
    """
    Generate a silent audio clip with the specified duration, fps, and number of channels.
    
    Parameters:
    - duration: The duration of the silence in seconds.
    - fps: The frames per second (sampling rate) of the audio.
    - n_channels: The number of audio channels (1 for mono, 2 for stereo, etc.).
    
    Returns:
    - A silent AudioClip of the specified duration and channels.
    """
    return AudioClip(lambda t: [0] * n_channels, duration=duration, fps=fps)

def analyze_and_adjust_audio(video_path, audio_path):
    """
    Analyze the video and audio lengths, adjust the audio file if necessary to ensure
    the audio length is less than or equal to the video length, and center the audio
    within the video duration.

    Parameters:
    - video_path: str, path to the input video file.
    - audio_path: str, path to the input audio file.

    Returns:
    - The path to the adjusted audio file.
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    video_duration = video_clip.duration
    audio_duration = audio_clip.duration

    # Calculate necessary padding to center the audio
    if audio_duration < video_duration:
        padding_duration = (video_duration - audio_duration) / 2
        # Generate silent audio clip for padding
        silent_clip = generate_silence(padding_duration, audio_clip.fps)
        # Prepend silent audio to original audio to center it
        centered_audio = concatenate_audioclips([silent_clip, audio_clip])

        # Save the adjusted (centered) audio
        adjusted_audio_path = os.path.splitext(audio_path)[0] + "-padded.mp3"
        centered_audio.write_audiofile(adjusted_audio_path)
    else:
        adjusted_audio_path = audio_path  # No change if audio is not shorter

    video_clip.close()
    audio_clip.close()

    return adjusted_audio_path

def run_wav2lip(checkpoint_path, video_path, audio_path, output_path):
    """
    Run the Wav2Lip model to lip-sync the video with the audio.

    Parameters:
    - checkpoint_path: str, path to the Wav2Lip checkpoint file.
    - video_path: str, path to the input video file.
    - audio_path: str, path to the input audio file.
    - output_path: str, path for the output video file.
    """
    # Construct the command to run the Wav2Lip model
    command = [
        'python', 'inference.py',
        '--checkpoint_path', checkpoint_path,
        '--face', video_path,
        '--audio', audio_path,
        '--outfile', output_path,
        '--face_det_batch_size', '32',
        '--wav2lip_batch_size', '256'
    ]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Successfully processed: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to process {video_path} with {audio_path}: {e}")

def main(folder_path, checkpoint_path, pad_audio=True):
    matches = find_matching_files(folder_path)
    for video_path, audio_path in matches:
        # Analyze and adjust audio
        if pad_audio:
            adjusted_audio_path = analyze_and_adjust_audio(video_path, audio_path)
        else:
            adjusted_audio_path = audio_path
        # Output filename
        output_path = os.path.splitext(video_path)[0] + "-output.mp4"
        # Run Wav2Lip
        run_wav2lip(checkpoint_path, video_path, adjusted_audio_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process videos and audios for lip syncing.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing video and audio files.")
    parser.add_argument("--pad_audio", type=str, default="True", help="Add silence to the beginning of the audio file to center within the video.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/wav2lip_gan.pth", help="Path to the Wav2Lip checkpoint file.")
    args = parser.parse_args()

    main(args.folder_path, args.checkpoint_path, args.pad_audio.lower() == "true")