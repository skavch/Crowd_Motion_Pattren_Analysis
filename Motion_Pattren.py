import cv2
import numpy as np
import streamlit as st
import tempfile
from io import BytesIO
import os

# Function to detect motion patterns (example from previous code)
def detect_motion_pattern(flow_vectors):
    magnitudes = np.linalg.norm(flow_vectors, axis=1)
    angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0]) * (180 / np.pi)
    
    avg_magnitude = np.mean(magnitudes)
    std_magnitude = np.std(magnitudes)

    if std_magnitude < 2 and avg_magnitude > 2:
        return "Laminar Flow"
    elif std_magnitude > 10:
        return "Turbulent Flow"
    elif np.mean(angles) > 45:
        return "Merging Flow"
    elif np.mean(angles) < -45:
        return "Diverging Flow"
    else:
        return "Crossing Flow"

# Function to process video with optical flow
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Define codec and create VideoWriter object to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output.name, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Parameters for Optical Flow
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Get the first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask for drawing
    mask = np.zeros_like(old_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2: Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        flow_vectors = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = int(new[0]), int(new[1])
            c, d = int(old[0]), int(old[1])
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
            flow_vectors.append([a - c, b - d])

        img = cv2.add(frame, mask)

        # Step 4: Detect motion pattern based on flow vectors
        if len(flow_vectors) > 0:
            flow_vectors = np.array(flow_vectors)
            motion_pattern = detect_motion_pattern(flow_vectors)

            # Display the detected motion pattern label on each frame
            cv2.putText(img, f'Motion Pattern: {motion_pattern}', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the frame to the video file
        out.write(img)

        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Release the capture and video writer
    cap.release()
    out.release()

    return temp_output.name  # Return path of the output video


# Streamlit UI 
st.set_page_config(page_title="Skavch Crowd Motion Pattren Analysis Engine", layout="wide")
# Add an image to the header
st.image("bg1.jpg", use_column_width=True)  # Adjust the image path as necessary
st.title("Skavch Crowd Motion Pattren Analysis Engine")

# Step 1: Upload video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temp file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input.close()

    # Process video
    st.text("Processing video...")
    processed_video_path = process_video(temp_input.name)
    st.text("Video processing completed!")

    # Step 2: Provide download link for processed video
    with open(processed_video_path, 'rb') as f:
        st.download_button(label="Download Processed Video", 
                           data=f, 
                           file_name="processed_crowd_motion.mp4", 
                           mime="video/mp4")

    # Cleanup: Remove temporary files
    os.remove(temp_input.name)
    os.remove(processed_video_path)
