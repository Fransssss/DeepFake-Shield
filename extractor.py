# extractor.py

# Importing the os module to handle file paths and folders
import os

# Importing OpenCV for working with video and images
import cv2

# tqdm helps show a progress bar when processing the video
from tqdm import tqdm

# This function takes a video and saves certain frames as images
def extract_frames_from_video(video_path, output_dir, fps_interval=1):
    """
    Extracts frames from a video at a specified frame-per-second (fps_interval).

    Args:
        video_path (str): Location of the video file.
        output_dir (str): Folder where frames will be saved.
        fps_interval (int): How often to grab a frame, in seconds.
    """
    # Create the output folder if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the video file
    cap = cv2.VideoCapture(video_path)

    # If the video can't be opened, show an error message and stop
    if not cap.isOpened():
        print(f"Oops, Failed to open video: {video_path}")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get how many frames the video shows every second (frames per second)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate how often to grab a frame based on the seconds we want
    frame_interval = video_fps * fps_interval

    # Start counting from the first frame
    frame_count = 0

    # Count how many frames we save
    saved_count = 0

    # Tell the user we're starting to extract frames
    print(f"Hi, we're extracting every {fps_interval}s from: {video_path} ...")

    # Set up the progress bar
    pbar = tqdm(total=total_frames)

    # Go through each frame in the video
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If there are no more frames, stop the loop
        if not ret:
            break

        # Only save the frame if it's at the right time interval
        if frame_count % frame_interval == 0:
            # Create a filename for the frame image
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")

            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)

            # Increase the count of saved frames
            saved_count += 1

        # Move to the next frame
        frame_count += 1

        # Update the progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()

    # Release the video file (free up memory)
    cap.release()

    # Tell the user how many frames were saved and where
    print(f"Done. Saved {saved_count} frames to {output_dir}")


# This part runs when the script is executed directly
if __name__ == "__main__":
    # Choose which video file to extract frames from
    sample_video = "video_samples/sample_1.mp4"

    # Get just the name of the video file (without extension)
    video_name = os.path.splitext(os.path.basename(sample_video))[0]

    # Create a folder to store the frames from that video
    output_folder = os.path.join("extracted_frames", video_name)

    # Call the function to extract frames
    extract_frames_from_video(sample_video, output_folder, fps_interval=1)


