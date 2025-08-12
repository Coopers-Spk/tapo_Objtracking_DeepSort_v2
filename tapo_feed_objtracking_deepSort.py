import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import Counter

class DeepSortTracker:
    def __init__(self, model_path='yolov8n.pt', rtsp_url=None):

        print(f"Loading YOLO model from {model_path}...")
        try:
            self.model = YOLO(model_path)
            print("YOLO model loaded successfully.")
        except Exception as e:
            raise ValueError(f"Error loading YOLO model from '{model_path}': {e}. "
                             "Please ensure the model path is correct and 'ultralytics' is installed.")
        
        print("Initializing DeepSORT tracker...")
        
        self.tracker = DeepSort(
            max_age=25,                  # Max frames before discard tracking
            n_init=5,                    # Min frames before track
            nms_max_overlap=1.0,         # 1,0 for deepSort
            max_cosine_distance=0.4,     # Max cosine distance for embed appear match (.2/.5)
            nn_budget=None,              # Maximum size of appearance descriptor gallery
            embedder="mobilenet",        # Appearance Embed generator can also use osnet
            half=True,                   # Faster if have right hardware - default True
            bgr=True,                    # BGR default OpenCV
            embedder_gpu=True,           # Use GPU for embedder - if CUDA
            polygon=False,               # Use for rotating objects
        )
        print("DeepSORT tracker initialized successfully with embedder (mobilenet).")
        
        # Initilize video feed (resolution and FPS hardset with tapo - change from feed 1 to 2)
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 8) # tapo is 15fps

        print(f"Video capture initialized. "
              f"Actual Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
              f"Actual FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        # Track statistics
        self.track_history = {}
        self.frame_count = 0
        self.id_history = []

        # Frame size (Used in draw_tracks)
        self.framewidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameheight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Tripline
        self.half_width = int(self.framewidth/2)
        self.tripline_tracker = 0
        self.tripline_id = []

        # Exclusions
        self.not_moved = {}

        # Text Size (auto scales between tapo feed 1 & 2)
        if self.framewidth == 640 and self.frameheight == 360:
            self.thick = 1
            self.scale = .3
        else:
            self.thick = 2
            self.scale = .7
        
        # Colors for different tracks (BGR format)
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (128, 0, 0),
            (255, 20, 147), (0, 191, 255), (255, 140, 0), (75, 0, 130)
        ]

    def process_detections(self, results):
        # Convert YOLO results to format needed by DeepSORT: (bbox_xywh, confidence, class_id)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy()) # Ensure float type
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter by confidence threshold to reduce noise for the tracker
                    if confidence > 0.5:
                        # Convert to [x, y, width, height] format for DeepSORT
                        bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        detections.append((bbox_xywh, confidence, class_id))
                        self.id_history.append(class_id)
        return detections

    # Meant to clean track memory for past tracks
    """def cleanup_inactive_tracks(self, tracks):
        #Removes inactive tracks from data structures to prevent memory waste in dict.
        active_track_ids = {int(t.track_id) for t in tracks}
        history_keys_to_check = list(self.track_history.keys()) 
        for track_id in history_keys_to_check:
            if track_id not in active_track_ids:
                del self.track_history[track_id]

    # Nightmare - drop objects that are not moving
    def still_objects(self):
        for track_id in self.track_history:
            if len(self.track_history[track_id]) >= 3:
                recent_points = self.track_history[track_id][-3:]
                x_values = [point[0] for point in recent_points[:-1]]  # All but last point
                avg_x = sum(x_values) / len(x_values)
                moved = abs(self.track_history[track_id][-1][0] - avg_x)
                if moved > 3:
                    self.not_moved[track_id] = True
                else:
                    self.not_moved[track_id] = False"""

    def draw_tracks(self, frame, tracks):
        """Draw tracking information on frame"""
        for track in tracks:
            if not track.is_confirmed():
                continue
            """
            print("Track sample:", dir(track))
            print("Has attrs:", hasattr(track, "track_id"), hasattr(track, "to_tlwh"), hasattr(track, "to_ltrb"))"""

            track_id = int(track.track_id)
            bbox = track.to_ltwh()  # left, top, width, height
            
            # Get class information if available
            class_id = track.get_det_class() if hasattr(track, 'get_det_class') else None
            confidence = track.get_det_conf() if hasattr(track, 'get_det_conf') else None
            
            # Convert bbox to integer coordinates
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]  # Frame is array, .shape returns (h,w, channels). [:2] returns first two
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            # Choose color based on track ID
            color = self.colors[track_id % len(self.colors)]
                               
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.thick)
            
            # Create label with available information
            label_parts = [f'ID: {track_id}']
            
            if class_id is not None and hasattr(self.model, 'names'):
                class_name = self.model.names.get(class_id, 'Unknown')
                label_parts.insert(0, class_name)
                
            if confidence is not None:
                label_parts.append(f'({confidence:.2f})') #keeps float to .00
                
            label = ' '.join(label_parts)
                
            # Calculate text size for background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.scale, self.thick)
            
            buffer = 3 # Buffer for rectangle around text
            text_y = max(text_height + buffer, y)  # Ensure text fits within frame
            cv2.rectangle(frame, (x, text_y - text_height - buffer),     # Draw background rectangle for text
                         (x + text_width, text_y + buffer), color, -1) 
            
            # Draw text
            cv2.putText(frame, label, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.scale, (255, 255, 255), self.thick)
            
            # Update track history for trajectory
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            # Store center point
            center = (x + w // 2, y + h // 2)
            self.track_history[track_id].append(center)
           
            # Tripline Tracker
            if track_id not in self.tripline_id:
                history = self.track_history[track_id]
                if len(history) >= 2:
                    previous_point = history[-2]
                    current_point = history[-1]

                    if (previous_point[0] < self.half_width and current_point[0] >= self.half_width) or \
                    (previous_point[0] > self.half_width and current_point[0] <= self.half_width):
                        self.tripline_tracker += 1
                        self.tripline_id.append(track_id)
            
            # Keep only last 30 points for trajectory
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
                #self.tripline_id[track_id].pop(0)
                    
            # Draw trajectory
            if len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
                
            # Draw center point
            cv2.circle(frame, center, 4, color, -1)

    def run(self):
        """Main tracking loop"""
        print("\nStarting DeepSORT object tracking... Press 'q' to quit")
        print("Note: Initial model loading might take a moment, and performance depends on hardware.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from video source. "
                      "This might indicate the stream ended, disconnected, or the webcam failed. Exiting...")
                break
                
            self.frame_count += 1
            
            results = self.model(frame, verbose=False, stream=True) 
    
            detections = self.process_detections(results)

            tracks = self.tracker.update_tracks(detections, frame=frame)

            #self.cleanup_inactive_tracks(tracks)
            self.still_objects(tracks)

            # Tripline RTSP = 127
            x = self.half_width 
            cv2.line(frame, (x, 0),(x, self.frameheight), (255,0,0), 1)
            
            # Draw tracking results on the frame
            self.draw_tracks(frame, tracks)
            
            # Add general frame information to the display. 't' is added to list for len()
            confirmed_tracks = len([t for t in tracks if t.is_confirmed()])
            total_detections = len(detections)

            # Set offset for info_text
            info_text = f"Frame: {self.frame_count} | Detections: {total_detections} | Active Tracks: {confirmed_tracks} | Count: {self.tripline_tracker}" 
            (top_text_w, top_text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, self.scale, self.thick)

            deepsort_info = f"Total Unique Tracks Created: {len(self.track_history)}"
            ((_,bottom_text_h), _) = cv2.getTextSize(deepsort_info, cv2.FONT_HERSHEY_DUPLEX, self.scale,self.thick)

            x_offset = int(self.framewidth - top_text_w - 4)
            y_offset = int(self.frameheight - top_text_h - bottom_text_h - 6)
            bs_text = y_offset + top_text_h + 3
            # Top Text
            cv2.putText(frame, info_text,(x_offset, y_offset), cv2.FONT_HERSHEY_DUPLEX, self.scale,(0, 255, 0), self.thick)
            # Bottom Text
            cv2.putText(frame, deepsort_info, (x_offset, bs_text), cv2.FONT_HERSHEY_DUPLEX, self.scale, (0, 255, 0), self.thick)

            # Display the annotated frame
            cv2.namedWindow("DeepSORT Object Tracking", cv2.WINDOW_NORMAL)            
            cv2.imshow('DeepSORT Object Tracking', frame)
            
            # Check for 'q' key press to quit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n'q' pressed. Exiting tracking loop.")
                break
        
        # Cleanup resources
        print("\nReleasing video capture and destroying windows...")
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Calculating totals
        '''
        counts = Counter(self.id_history)
        for class_id in counts:
            quant = counts[class_id]
            name = self.model.names.get(class_id, 'Unknown')
            print(name,quant)
            '''
        # Print final statistics
        print(f"\n--- DeepSORT Tracking Session Statistics ---")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total unique tracks created (ever seen): {len(self.track_history)}")
        if self.track_history:
            avg_track_length = np.mean([len(history) for history in self.track_history.values()])
            print(f"Average track length: {avg_track_length:.1f} frames")
        else:
            print("No tracks were created or confirmed during this session.")
        print(f"--------------------------------------------")

def main():
    # --- Configuration ---
    # Choose your YOLO model. n,s,m,
    MODEL_PATH = 'yolov8m.pt' 

    # Change to here to RTSP stream 1 for high res, but will run very slow
    RTSP_URL = "------Insert RTSP here-------"
    
    print("\n--- Starting DeepSORT Tracker Application ---")
    try:
        # Create tracker instance
        tracker = DeepSortTracker(model_path=MODEL_PATH, rtsp_url=RTSP_URL)
        
        # Start the main tracking loop
        tracker.run()
    except ValueError as ve:
        print(f"\nApplication Setup Error: {ve}")
        print("Please review your MODEL_PATH, RTSP_URL, and webcam setup.")
    except Exception as e:
        print(f"\nAn unexpected runtime error occurred: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
    finally:
        print("\n--- DeepSORT Tracker Application Ended ---")

if __name__ == "__main__":

    main()
