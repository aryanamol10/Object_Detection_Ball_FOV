#!/usr/bin/env python3
import cv2
import numpy as np
import os
from datetime import datetime
import argparse
import json
import time

DEFAULT_BOARD_SIZE = (11, 8)
DEFAULT_SQUARE_SIZE = 0.045
DEFAULT_MAX_IMAGES = 100
DEFAULT_OUTPUT_DIR = "calibration_images"

class CameraCalibrator:
    def __init__(self, board_size=DEFAULT_BOARD_SIZE, square_size=DEFAULT_SQUARE_SIZE,
                 mode="chessboard", output_dir="calibration_images", 
                 use_initial_guess=True, continue_from_existing=False):
        self.board_size = board_size
        self.square_size = square_size
        self.mode = mode.lower()
        self.marker_size = square_size * 0.8
        self.output_dir = output_dir
        self.use_initial_guess = use_initial_guess

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        os.makedirs(self.output_dir, exist_ok=True)

        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

        self.obj_points = []
        self.img_points = []

        # Load existing calibration images if continuing
        if continue_from_existing:
            self._load_existing_images()

        # Intrinsics from prior calibration (only if not blank)
        if use_initial_guess:
            self.base_camera_matrix = np.array([
                [891.52846853, 0.0, 640.75445056],
                [0.0, 891.74414996, 413.9437101],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            self.base_dist_coeffs = np.array(
                [[-0.41094911, 0.209865385, -0.000119377526, -3.45794783e-05, -0.0582352254]],
                dtype=np.float64
            )
        else:
            self.base_camera_matrix = None
            self.base_dist_coeffs = None

    def _load_existing_images(self):
        """Load existing calibration images from the output directory and re-detect corners."""
        if not os.path.exists(self.output_dir):
            print(f"Output directory {self.output_dir} does not exist. Starting fresh.")
            return
        
        # Find all calibration images
        image_files = [f for f in os.listdir(self.output_dir) 
                      if f.startswith('calib_') and f.endswith('.png')]
        
        if not image_files:
            print("No existing calibration images found. Starting fresh.")
            return
        
        print(f"Loading {len(image_files)} existing calibration images...")
        loaded_count = 0
        
        for img_file in sorted(image_files):
            img_path = os.path.join(self.output_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Re-detect corners
            found, corners = self.detect_corners(img)
            
            if found:
                self.obj_points.append(self.objp.copy())
                self.img_points.append(corners.reshape(-1, 1, 2).astype(np.float32))
                loaded_count += 1
        
        print(f"Successfully loaded {loaded_count} existing calibration images.")

    def detect_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, self.board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret and corners is not None:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners2
        return False, None

    def save_image(self, img, corners):
        cv2.drawChessboardCorners(img, self.board_size, corners, True)
        self.obj_points.append(self.objp.copy())
        self.img_points.append(corners.reshape(-1, 1, 2).astype(np.float32))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.output_dir, f"calib_{timestamp}.png")
        cv2.imwrite(filename, img)
        print(f"Saved: {filename}")
        return len(self.obj_points)

    def calibrate(self, image_size):
        if len(self.obj_points) < 5:
            print("Not enough calibration images. Need at least 5.")
            return None, None, None, None, None

        print(f"Calibrating camera with {len(self.obj_points)} images...")

        flags = 0
        initial_matrix = None
        initial_dist = None
        
        if self.use_initial_guess and self.base_camera_matrix is not None:
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
            initial_matrix = self.base_camera_matrix.copy()
            initial_dist = self.base_dist_coeffs.copy() if self.base_dist_coeffs is not None else None
        
        ret, mtx, dist, rvecs, tvecs, stdDevsIntrinsics, stdDevsExtrinsics, perViewErrors = \
            cv2.calibrateCameraExtended(
                self.obj_points, self.img_points, image_size,
                cameraMatrix=initial_matrix,
                distCoeffs=initial_dist,
                flags=flags
            )

        # Reject high-error outliers (frames > 2× median)
        perViewErrors = perViewErrors.flatten()
        median_err = np.median(perViewErrors)
        threshold = 2.0 * median_err
        keep_idx = [i for i, e in enumerate(perViewErrors) if e <= threshold]

        if len(keep_idx) < len(self.obj_points):
            print(f"Rejected {len(self.obj_points) - len(keep_idx)} high-error frames.")
            self.obj_points = [self.obj_points[i] for i in keep_idx]
            self.img_points = [self.img_points[i] for i in keep_idx]
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, image_size,
                cameraMatrix=mtx, distCoeffs=dist, flags=flags
            )

        total_error, total_points = 0.0, 0
        for i in range(len(self.obj_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.obj_points[i], rvecs[i], tvecs[i], mtx, dist
            )
            err = cv2.norm(self.img_points[i], imgpoints2, cv2.NORM_L2)
            total_error += err ** 2
            total_points += len(imgpoints2)
        mean_error = np.sqrt(total_error / total_points)

        print(f"Calibration complete!\nReprojection RMSE: {mean_error:.6f}")
        print(f"Per-view median error: {median_err:.6f}")
        return mtx, dist, rvecs, tvecs, mean_error


def main():
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--width', type=int, default=11)
    parser.add_argument('--height', type=int, default=8)
    parser.add_argument('--square_size', type=float, default=0.060)
    parser.add_argument('--max_images', type=int, default=100)
    parser.add_argument('--output_name', type=str, default='calibration_images')
    parser.add_argument('--blank', action='store_true', 
                       help='Start calibration with no initial assumption (no prior camera matrix)')
    parser.add_argument('--c', '--continue', dest='continue_existing', action='store_true',
                       help='Continue calibration from existing calibration_images directory')
    args = parser.parse_args()

    board_size = (args.width, args.height)
    use_initial_guess = not args.blank
    calibrator = CameraCalibrator(
        board_size, args.square_size, "chessboard", args.output_name,
        use_initial_guess=use_initial_guess,
        continue_from_existing=args.continue_existing
    )

    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    

    # Start count from existing images if continuing
    saved_count = len(calibrator.obj_points)
    last_capture_time = 0
    min_interval = 1.5

    mode_str = "blank slate" if args.blank else ("continuing" if args.continue_existing else "with initial guess")
    print(f"Camera calibration tool — {mode_str}. Capturing automatically. Press 'q' to quit.")
    if args.continue_existing:
        print(f"Starting with {saved_count} existing images.")

    w = h = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        display = frame.copy()
        h, w = frame.shape[:2]
        found, corners = calibrator.detect_corners(frame)

        if found:
            corners_arr = corners.reshape(-1, 2)
            x_span = corners_arr[:, 0].max() - corners_arr[:, 0].min()
            y_span = corners_arr[:, 1].max() - corners_arr[:, 1].min()
            if x_span > 0.5 * w and y_span > 0.5 * h:
                if (time.time() - last_capture_time) > min_interval:
                    calibrator.save_image(frame.copy(), corners)
                    print(f"Auto-saved frame #{len(calibrator.obj_points)}")
                    last_capture_time = time.time()
                    if len(calibrator.obj_points) >= args.max_images:
                        print(f"Reached maximum of {args.max_images} images.")
                        break

        cv2.putText(display, f"Images saved: {len(calibrator.obj_points)}/{args.max_images}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera Calibration', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    total_images = len(calibrator.obj_points)
    if total_images >= 5 and w and h:
        mtx, dist, rvecs, tvecs, mean_error = calibrator.calibrate((w, h))
        if mtx is not None:
            calibration_data = {
                'camera_matrix': mtx.tolist(),
                'dist_coeffs': dist.tolist(),
                'rvecs': [r.tolist() for r in rvecs],
                'tvecs': [t.tolist() for t in tvecs],
                'board_size': board_size,
                'square_size': args.square_size,
                'image_size': [w, h],
                'num_images': total_images,
                'reprojection_error': float(mean_error)
            }

            np.savez(f"{args.output_name}.npz", **calibration_data)
            with open(f"{args.output_name}.json", 'w') as f:
                json.dump(calibration_data, f, indent=2)

            print(f"\nSaved calibration to {args.output_name}.npz and .json")
            print(f"Final RMSE: {mean_error:.6f}")
    else:
        print("Not enough images for calibration (need ≥5).")

if __name__ == "__main__":
    main()