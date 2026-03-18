import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class UnifiedVideoKeypointExtractor:
    def __init__(self, static_image_mode=False, model_complexity=2,
                 smooth_landmarks=True, refine_face_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_video_metadata(self, video_path):
        """Extract video information"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return metadata
    
    def extract_frame_landmarks(self, frame):
        """
        Extract all landmarks from a single frame
        Returns serializable dictionary
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.holistic.process(frame_rgb)
        
        landmarks_dict = {
            'face': [],
            'pose': [],
            'left_hand': [],
            'right_hand': [],
            'pose_world': []
        }
        
        # Face landmarks (468 points)
        if results.face_landmarks:
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                landmarks_dict['face'].append({
                    'idx': idx,
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(getattr(landmark, 'visibility', 1.0))
                })
        else:
            # If no face detected, include empty structure
            landmarks_dict['face'] = []
        
        # Pose landmarks (33 points)
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_dict['pose'].append({
                    'idx': idx,
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(landmark.visibility)
                })
        else:
            landmarks_dict['pose'] = []
        
        # Left hand landmarks (21 points)
        if results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                landmarks_dict['left_hand'].append({
                    'idx': idx,
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(getattr(landmark, 'visibility', 1.0))
                })
        else:
            landmarks_dict['left_hand'] = []
        
        # Right hand landmarks (21 points)
        if results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                landmarks_dict['right_hand'].append({
                    'idx': idx,
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(getattr(landmark, 'visibility', 1.0))
                })
        else:
            landmarks_dict['right_hand'] = []
        
        # Pose world landmarks (33 points in world coordinates)
        if results.pose_world_landmarks:
            for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
                landmarks_dict['pose_world'].append({
                    'idx': idx,
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(landmark.visibility)
                })
        else:
            landmarks_dict['pose_world'] = []
        
        return landmarks_dict
    
    def process_video_to_single_json(self, video_path, output_dir='video_keypoints_json'):
        """
        Process entire video and save ALL frames to a single JSON file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video metadata
        video_info = self.extract_video_metadata(video_path)
        if not video_info:
            print(f"Error: Could not read video {video_path}")
            return None
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_json_path = os.path.join(output_dir, f"{video_name}_keypoints.json")
        
        print(f"Processing: {video_name}")
        print(f"FPS: {video_info['fps']}, Total frames: {video_info['total_frames']}")
        print(f"Duration: {video_info['duration']:.2f} seconds")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        all_frames_data = {
            'video_info': video_info,
            'frames': []
        }
        
        frame_count = 0
        pbar = tqdm(total=video_info['total_frames'], desc=f"Processing {video_name}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract landmarks for this frame
            landmarks_dict = self.extract_frame_landmarks(frame)
            
            # Create frame data structure
            frame_data = {
                'frame_number': frame_count,
                'timestamp': float(frame_count / video_info['fps']),
                'landmarks': landmarks_dict,
                'detection_summary': {
                    'has_face': len(landmarks_dict['face']) > 0,
                    'has_pose': len(landmarks_dict['pose']) > 0,
                    'has_left_hand': len(landmarks_dict['left_hand']) > 0,
                    'has_right_hand': len(landmarks_dict['right_hand']) > 0
                }
            }
            
            all_frames_data['frames'].append(frame_data)
            
            frame_count += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        # Save ALL frames to a single JSON file
        print(f"Saving to JSON: {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(all_frames_data, f, indent=2)
        
        print(f"✅ Video processing complete!")
        print(f"📊 Total frames processed: {frame_count}")
        print(f"📁 JSON file saved: {output_json_path}")
        
        return output_json_path, all_frames_data


def process_all_videos_to_json(dataset_path, output_dir='all_videos_keypoints_json'):
    """
    Process ALL videos in dataset, each to its own JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    video_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} videos to process")
    
    # Initialize extractor
    extractor = UnifiedVideoKeypointExtractor(
        static_image_mode=False,  # False for video (enables tracking)
        model_complexity=2,       # Highest accuracy
        refine_face_landmarks=True
    )
    
    summary_data = []
    all_json_paths = []
    
    # Process each video
    for video_path in video_files:
        try:
            print(f"\n{'='*60}")
            video_name = os.path.basename(video_path)
            print(f"Processing: {video_name}")
            
            # Process video to single JSON
            json_path, video_data = extractor.process_video_to_single_json(video_path, output_dir)
            
            if json_path and video_data:
                all_json_paths.append(json_path)
                
                # Calculate statistics
                frames = video_data['frames']
                total_frames = len(frames)
                
                if total_frames > 0:
                    frames_with_face = sum(1 for f in frames if f['detection_summary']['has_face'])
                    frames_with_pose = sum(1 for f in frames if f['detection_summary']['has_pose'])
                    frames_with_left_hand = sum(1 for f in frames if f['detection_summary']['has_left_hand'])
                    frames_with_right_hand = sum(1 for f in frames if f['detection_summary']['has_right_hand'])
                    
                    summary_data.append({
                        'video_name': video_name,
                        'json_file': os.path.basename(json_path),
                        'total_frames': total_frames,
                        'duration': video_data['video_info']['duration'],
                        'fps': video_data['video_info']['fps'],
                        'width': video_data['video_info']['width'],
                        'height': video_data['video_info']['height'],
                        'frames_with_face': frames_with_face,
                        'frames_with_pose': frames_with_pose,
                        'frames_with_left_hand': frames_with_left_hand,
                        'frames_with_right_hand': frames_with_right_hand,
                        'face_percentage': (frames_with_face / total_frames) * 100,
                        'pose_percentage': (frames_with_pose / total_frames) * 100,
                        'left_hand_percentage': (frames_with_left_hand / total_frames) * 100,
                        'right_hand_percentage': (frames_with_right_hand / total_frames) * 100
                    })
                
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    # Save summary CSV
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'processing_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print("✅ ALL VIDEOS PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"📊 Total videos processed: {len(summary_data)}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📈 Summary saved to: {summary_path}")
        
        # Print overall statistics
        print(f"\n📊 Overall Statistics:")
        print(f"Total videos: {len(summary_data)}")
        print(f"Total frames processed: {df_summary['total_frames'].sum()}")
        print(f"Total duration: {df_summary['duration'].sum():.2f} seconds")
        print(f"\nAverage Detection Rates:")
        print(f"  Face: {df_summary['face_percentage'].mean():.1f}%")
        print(f"  Pose: {df_summary['pose_percentage'].mean():.1f}%")
        print(f"  Left Hand: {df_summary['left_hand_percentage'].mean():.1f}%")
        print(f"  Right Hand: {df_summary['right_hand_percentage'].mean():.1f}%")
        
        return all_json_paths, df_summary
    
    return None, None


def load_video_json(json_path):
    """Load a video JSON file and return data"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_keypoint_statistics(json_path):
    """Extract statistics from a video JSON file"""
    data = load_video_json(json_path)
    
    frames = data['frames']
    video_info = data['video_info']
    
    stats = {
        'video_name': os.path.basename(json_path).replace('_keypoints.json', ''),
        'total_frames': len(frames),
        'duration': video_info['duration'],
        'fps': video_info['fps']
    }
    
    # Detection rates
    detection_fields = ['has_face', 'has_pose', 'has_left_hand', 'has_right_hand']
    
    for field in detection_fields:
        count = sum(1 for f in frames if f['detection_summary'][field])
        stats[f'{field}_count'] = count
        stats[f'{field}_percentage'] = (count / len(frames)) * 100
    
    # Keypoint position statistics
    all_keypoints = []
    for frame in frames:
        for landmark_type in ['face', 'pose', 'left_hand', 'right_hand']:
            if frame['landmarks'][landmark_type]:
                for point in frame['landmarks'][landmark_type]:
                    all_keypoints.append({
                        'x': point['x'],
                        'y': point['y'],
                        'z': point['z']
                    })
    
    if all_keypoints:
        all_keypoints_df = pd.DataFrame(all_keypoints)
        stats['avg_x_position'] = all_keypoints_df['x'].mean()
        stats['avg_y_position'] = all_keypoints_df['y'].mean()
        stats['avg_z_position'] = all_keypoints_df['z'].mean()
    
    return stats


def create_unified_dataset_json(all_json_paths, output_file='unified_dataset.json'):
    """
    Combine ALL video JSON files into one unified JSON file
    Warning: This can be very large!
    """
    unified_data = {
        'dataset_info': {
            'total_videos': len(all_json_paths),
            'creation_date': pd.Timestamp.now().isoformat()
        },
        'videos': []
    }
    
    for json_path in tqdm(all_json_paths, desc="Combining JSON files"):
        try:
            with open(json_path, 'r') as f:
                video_data = json.load(f)
            
            video_name = os.path.basename(json_path).replace('_keypoints.json', '')
            
            unified_data['videos'].append({
                'video_name': video_name,
                'file_path': json_path,
                'data': video_data
            })
            
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue
    
    # Save unified JSON
    with open(output_file, 'w') as f:
        json.dump(unified_data, f, indent=2)
    
    print(f"✅ Unified dataset saved to: {output_file}")
    print(f"📊 Total videos included: {len(unified_data['videos'])}")
    
    return unified_data


# Example usage and visualization functions
if __name__ == "__main__":
    # Set your dataset path
    dataset_path = "dataset1"  # Folder containing videos
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path '{dataset_path}' does not exist!")
        print(f"Current directory: {os.getcwd()}")
        print("Available folders:", os.listdir('.'))
    else:
        print(f"📁 Processing video dataset from: {dataset_path}")
        print(f"📊 Folder contents: {os.listdir(dataset_path)}")
        
        # Process all videos to individual JSON files
        json_paths, summary_df = process_all_videos_to_json(
            dataset_path, 
            output_dir='video_keypoints_json_output1'
        )
        
        # Example: Load and analyze a specific video's JSON
        if json_paths:
            print(f"\n📋 Example JSON structure for first video:")
            first_video_data = load_video_json(json_paths[0])
            
            print(f"Video: {os.path.basename(json_paths[0])}")
            print(f"Total frames: {len(first_video_data['frames'])}")
            print(f"First frame keys: {list(first_video_data['frames'][0].keys())}")
            
            # Show sample frame data
            sample_frame = first_video_data['frames'][0]
            print(f"\nSample Frame 0:")
            print(f"  Timestamp: {sample_frame['timestamp']:.2f}s")
            print(f"  Face points: {len(sample_frame['landmarks']['face'])}")
            print(f"  Pose points: {len(sample_frame['landmarks']['pose'])}")
            print(f"  Left hand points: {len(sample_frame['landmarks']['left_hand'])}")
            print(f"  Right hand points: {len(sample_frame['landmarks']['right_hand'])}")
            
            # Extract statistics
            stats = extract_keypoint_statistics(json_paths[0])
            print(f"\n📈 Video Statistics:")
            for key, value in stats.items():
                if 'percentage' in key:
                    print(f"  {key}: {value:.1f}%")
                else:
                    print(f"  {key}: {value}")
            
            # Optional: Create unified dataset (caution: can be very large!)
            # unified_data = create_unified_dataset_json(json_paths, 'all_videos_unified.json')