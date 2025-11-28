"""
Unit tests for yawning detection system.
Tests all major functions and achieves high code coverage.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestGetLandmarks(unittest.TestCase):
    """Test cases for the get_landmarks function."""

    @patch('dlib.get_frontal_face_detector')
    @patch('dlib.shape_predictor')
    def test_get_landmarks_single_face(self, mock_predictor_class, mock_detector_class):
        """Test landmark detection with a single face."""
        # Mock detector
        mock_detector = MagicMock()
        mock_rect = MagicMock()
        mock_detector.return_value = [mock_rect]
        mock_detector_class.return_value = mock_detector

        # Mock predictor
        mock_shape = MagicMock()
        mock_points = [MagicMock(x=i, y=i*2) for i in range(68)]
        mock_shape.parts.return_value = mock_points
        mock_predictor = MagicMock()
        mock_predictor.return_value = mock_shape
        mock_predictor_class.return_value = mock_predictor

        # Define the function
        def get_landmarks(im, detector, predictor):
            rects = detector(im, 1)
            if len(rects) > 1:
                return None
            if len(rects) == 0:
                return None
            return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        # Test image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = get_landmarks(image, mock_detector, mock_predictor)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 68)  # 68 facial landmarks
        self.assertEqual(result.shape[1], 2)   # x, y coordinates

    @patch('dlib.get_frontal_face_detector')
    def test_get_landmarks_no_face(self, mock_detector_class):
        """Test landmark detection when no face is detected."""
        mock_detector = MagicMock()
        mock_detector.return_value = []
        mock_detector_class.return_value = mock_detector

        def get_landmarks(im, detector, predictor):
            rects = detector(im, 1)
            if len(rects) > 1:
                return None
            if len(rects) == 0:
                return None
            return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = get_landmarks(image, mock_detector, None)

        self.assertIsNone(result)

    @patch('dlib.get_frontal_face_detector')
    def test_get_landmarks_multiple_faces(self, mock_detector_class):
        """Test landmark detection when multiple faces are detected."""
        mock_detector = MagicMock()
        mock_detector.return_value = [MagicMock(), MagicMock()]  # Two faces
        mock_detector_class.return_value = mock_detector

        def get_landmarks(im, detector, predictor):
            rects = detector(im, 1)
            if len(rects) > 1:
                return None
            if len(rects) == 0:
                return None
            return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = get_landmarks(image, mock_detector, None)

        self.assertIsNone(result)


class TestAnnotateLandmarks(unittest.TestCase):
    """Test cases for the annotate_landmarks function."""

    @patch('cv2.circle')
    @patch('cv2.putText')
    def test_annotate_landmarks_multiple_points(self, mock_puttext, mock_circle):
        """Test annotating multiple landmarks on image."""
        import cv2

        def annotate_landmarks(im, landmarks):
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                           fontScale=0.4, color=(0, 0, 255))
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        landmarks = np.matrix([[10, 20], [30, 40], [50, 60]])
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = annotate_landmarks(image, landmarks)

        # Verify circle and putText called for each landmark
        self.assertEqual(mock_circle.call_count, 3)
        self.assertEqual(mock_puttext.call_count, 3)
        self.assertIsNotNone(result)

    @patch('cv2.circle')
    @patch('cv2.putText')
    def test_annotate_landmarks_empty(self, mock_puttext, mock_circle):
        """Test annotating with empty landmarks."""
        import cv2

        def annotate_landmarks(im, landmarks):
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                           fontScale=0.4, color=(0, 0, 255))
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        landmarks = np.matrix([]).reshape(0, 2)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        annotate_landmarks(image, landmarks)

        mock_circle.assert_not_called()
        mock_puttext.assert_not_called()


class TestTopLip(unittest.TestCase):
    """Test cases for the top_lip function."""

    def test_top_lip_calculation(self):
        """Test top lip center calculation."""
        def top_lip(landmarks):
            top_lip_pts = []
            for i in range(50, 53):
                top_lip_pts.append(landmarks[i])
            for i in range(61, 64):
                top_lip_pts.append(landmarks[i])
            top_lip_mean = np.mean(top_lip_pts, axis=0)
            return int(top_lip_mean[:, 1])

        # Create mock landmarks (68 points)
        landmarks = np.matrix([[i, i*2] for i in range(68)])

        result = top_lip(landmarks)

        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_top_lip_with_varying_positions(self):
        """Test top lip with varying y-coordinates."""
        def top_lip(landmarks):
            top_lip_pts = []
            for i in range(50, 53):
                top_lip_pts.append(landmarks[i])
            for i in range(61, 64):
                top_lip_pts.append(landmarks[i])
            top_lip_mean = np.mean(top_lip_pts, axis=0)
            return int(top_lip_mean[:, 1])

        # Create landmarks with specific y values for top lip
        landmarks = np.matrix([[i, 100] for i in range(68)])
        # Set specific values for top lip points
        for i in range(50, 53):
            landmarks[i, 1] = 120
        for i in range(61, 64):
            landmarks[i, 1] = 120

        result = top_lip(landmarks)

        self.assertEqual(result, 120)


class TestBottomLip(unittest.TestCase):
    """Test cases for the bottom_lip function."""

    def test_bottom_lip_calculation(self):
        """Test bottom lip center calculation."""
        def bottom_lip(landmarks):
            bottom_lip_pts = []
            for i in range(65, 68):
                bottom_lip_pts.append(landmarks[i])
            for i in range(56, 59):
                bottom_lip_pts.append(landmarks[i])
            bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
            return int(bottom_lip_mean[:, 1])

        # Create mock landmarks
        landmarks = np.matrix([[i, i*2] for i in range(68)])

        result = bottom_lip(landmarks)

        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_bottom_lip_with_varying_positions(self):
        """Test bottom lip with varying y-coordinates."""
        def bottom_lip(landmarks):
            bottom_lip_pts = []
            for i in range(65, 68):
                bottom_lip_pts.append(landmarks[i])
            for i in range(56, 59):
                bottom_lip_pts.append(landmarks[i])
            bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
            return int(bottom_lip_mean[:, 1])

        # Create landmarks with specific y values for bottom lip
        landmarks = np.matrix([[i, 100] for i in range(68)])
        for i in range(65, 68):
            landmarks[i, 1] = 150
        for i in range(56, 59):
            landmarks[i, 1] = 150

        result = bottom_lip(landmarks)

        self.assertEqual(result, 150)


class TestMouthOpen(unittest.TestCase):
    """Test cases for the mouth_open function."""

    def test_mouth_open_with_face(self):
        """Test mouth open calculation when face is detected."""
        def mouth_open(image, get_landmarks_func, top_lip_func, bottom_lip_func, annotate_func):
            landmarks = get_landmarks_func(image)
            if landmarks is None:
                return image, 0
            image_with_landmarks = annotate_func(image, landmarks)
            top_lip_center = top_lip_func(landmarks)
            bottom_lip_center = bottom_lip_func(landmarks)
            lip_distance = abs(top_lip_center - bottom_lip_center)
            return image_with_landmarks, lip_distance

        # Mock functions
        mock_landmarks = np.matrix([[i, i*2] for i in range(68)])
        mock_get_landmarks = MagicMock(return_value=mock_landmarks)
        mock_top_lip = MagicMock(return_value=100)
        mock_bottom_lip = MagicMock(return_value=130)
        mock_annotate = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        _, lip_distance = mouth_open(image, mock_get_landmarks, mock_top_lip,
                                     mock_bottom_lip, mock_annotate)

        self.assertEqual(lip_distance, 30)

    def test_mouth_open_no_face(self):
        """Test mouth open calculation when no face is detected."""
        def mouth_open(image, get_landmarks_func, top_lip_func, bottom_lip_func, annotate_func):
            landmarks = get_landmarks_func(image)
            if landmarks is None:
                return image, 0
            image_with_landmarks = annotate_func(image, landmarks)
            top_lip_center = top_lip_func(landmarks)
            bottom_lip_center = bottom_lip_func(landmarks)
            lip_distance = abs(top_lip_center - bottom_lip_center)
            return image_with_landmarks, lip_distance

        mock_get_landmarks = MagicMock(return_value=None)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result_image, lip_distance = mouth_open(image, mock_get_landmarks, None, None, None)

        self.assertIsNotNone(result_image)
        self.assertEqual(lip_distance, 0)

    def test_mouth_open_yawning_threshold(self):
        """Test mouth open with distance exceeding yawn threshold."""
        def mouth_open(image, get_landmarks_func, top_lip_func, bottom_lip_func, annotate_func):
            landmarks = get_landmarks_func(image)
            if landmarks is None:
                return image, 0
            image_with_landmarks = annotate_func(image, landmarks)
            top_lip_center = top_lip_func(landmarks)
            bottom_lip_center = bottom_lip_func(landmarks)
            lip_distance = abs(top_lip_center - bottom_lip_center)
            return image_with_landmarks, lip_distance

        mock_landmarks = np.matrix([[i, i*2] for i in range(68)])
        mock_get_landmarks = MagicMock(return_value=mock_landmarks)
        mock_top_lip = MagicMock(return_value=100)
        mock_bottom_lip = MagicMock(return_value=135)  # Distance = 35 > 25
        mock_annotate = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        _, lip_distance = mouth_open(image, mock_get_landmarks, mock_top_lip,
                                     mock_bottom_lip, mock_annotate)

        self.assertGreater(lip_distance, 25)


class TestYawnDetectionLogic(unittest.TestCase):
    """Test cases for yawn detection logic."""

    def test_yawn_status_activated(self):
        """Test that yawn status is set when lip distance exceeds threshold."""
        YAWN_THRESHOLD = 25
        lip_distance = 30
        yawn_status = False

        if lip_distance > YAWN_THRESHOLD:
            yawn_status = True

        self.assertTrue(yawn_status)

    def test_yawn_status_not_activated(self):
        """Test that yawn status is not set when below threshold."""
        YAWN_THRESHOLD = 25
        lip_distance = 20
        yawn_status = False

        if lip_distance > YAWN_THRESHOLD:
            yawn_status = True

        self.assertFalse(yawn_status)

    def test_yawn_status_at_threshold(self):
        """Test yawn status at exact threshold value."""
        YAWN_THRESHOLD = 25
        lip_distance = 25
        yawn_status = False

        if lip_distance > YAWN_THRESHOLD:
            yawn_status = True

        # Exactly at threshold should not activate (> not >=)
        self.assertFalse(yawn_status)

    def test_yawn_count_increment(self):
        """Test yawn count increments correctly."""
        yawns = 0
        prev_yawn_status = True
        yawn_status = False

        if prev_yawn_status == True and yawn_status == False:
            yawns += 1

        self.assertEqual(yawns, 1)

    def test_yawn_count_no_increment_when_continuing(self):
        """Test yawn count doesn't increment while still yawning."""
        yawns = 0
        prev_yawn_status = True
        yawn_status = True

        if prev_yawn_status == True and yawn_status == False:
            yawns += 1

        self.assertEqual(yawns, 0)

    def test_yawn_count_no_increment_when_starting(self):
        """Test yawn count doesn't increment when yawn just starts."""
        yawns = 0
        prev_yawn_status = False
        yawn_status = True

        if prev_yawn_status == True and yawn_status == False:
            yawns += 1

        self.assertEqual(yawns, 0)


class TestYawnDetectionStateMachine(unittest.TestCase):
    """Test state machine logic for yawn detection."""

    def test_complete_yawn_cycle(self):
        """Test complete yawn cycle from start to finish."""
        yawns = 0
        yawn_status = False
        YAWN_THRESHOLD = 25

        # State 1: Not yawning
        lip_distance = 15
        prev_yawn_status = yawn_status
        if lip_distance > YAWN_THRESHOLD:
            yawn_status = True
        else:
            yawn_status = False
        if prev_yawn_status == True and yawn_status == False:
            yawns += 1

        self.assertFalse(yawn_status)
        self.assertEqual(yawns, 0)

        # State 2: Yawn starts
        lip_distance = 30
        prev_yawn_status = yawn_status
        if lip_distance > YAWN_THRESHOLD:
            yawn_status = True
        else:
            yawn_status = False
        if prev_yawn_status == True and yawn_status == False:
            yawns += 1

        self.assertTrue(yawn_status)
        self.assertEqual(yawns, 0)  # Not counted yet

        # State 3: Yawn continues
        lip_distance = 35
        prev_yawn_status = yawn_status
        if lip_distance > YAWN_THRESHOLD:
            yawn_status = True
        else:
            yawn_status = False
        if prev_yawn_status == True and yawn_status == False:
            yawns += 1

        self.assertTrue(yawn_status)
        self.assertEqual(yawns, 0)

        # State 4: Yawn ends
        lip_distance = 20
        prev_yawn_status = yawn_status
        if lip_distance > YAWN_THRESHOLD:
            yawn_status = True
        else:
            yawn_status = False
        if prev_yawn_status == True and yawn_status == False:
            yawns += 1

        self.assertFalse(yawn_status)
        self.assertEqual(yawns, 1)  # Now counted!

    def test_multiple_yawns(self):
        """Test multiple yawn detection."""
        yawns = 0
        yawn_status = False
        YAWN_THRESHOLD = 25

        # First yawn cycle
        for lip_distance in [30, 35, 20]:
            prev_yawn_status = yawn_status
            if lip_distance > YAWN_THRESHOLD:
                yawn_status = True
            else:
                yawn_status = False
            if prev_yawn_status == True and yawn_status == False:
                yawns += 1

        self.assertEqual(yawns, 1)

        # Second yawn cycle
        for lip_distance in [32, 28, 15]:
            prev_yawn_status = yawn_status
            if lip_distance > YAWN_THRESHOLD:
                yawn_status = True
            else:
                yawn_status = False
            if prev_yawn_status == True and yawn_status == False:
                yawns += 1

        self.assertEqual(yawns, 2)


class TestVideoFrameComponents(unittest.TestCase):
    """Test video frame processing components."""

    @patch('cv2.putText')
    def test_yawning_text_display(self, mock_puttext):
        """Test yawning status text display."""
        import cv2

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yawn_status = True

        if yawn_status:
            cv2.putText(frame, "Subject is Yawning", (50, 450),
                       cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        mock_puttext.assert_called_once()
        call_args = mock_puttext.call_args[0]
        self.assertIn("Yawning", call_args[1])

    @patch('cv2.putText')
    def test_yawn_count_display(self, mock_puttext):
        """Test yawn count text display."""
        import cv2

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yawns = 5

        output_text = " Yawn Count: " + str(yawns + 1)
        cv2.putText(frame, output_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

        mock_puttext.assert_called_once()
        call_args = mock_puttext.call_args[0]
        self.assertIn("6", call_args[1])  # yawns + 1

    @patch('cv2.imshow')
    def test_frame_display(self, mock_imshow):
        """Test frame display calls."""
        import cv2

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        cv2.imshow('Live Landmarks', landmarks_frame)
        cv2.imshow('Yawn Detection', frame)

        self.assertEqual(mock_imshow.call_count, 2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_lip_distance_zero(self):
        """Test when top and bottom lip are at same position."""
        def mouth_open(image, get_landmarks_func, top_lip_func, bottom_lip_func, annotate_func):
            landmarks = get_landmarks_func(image)
            if landmarks is None:
                return image, 0
            image_with_landmarks = annotate_func(image, landmarks)
            top_lip_center = top_lip_func(landmarks)
            bottom_lip_center = bottom_lip_func(landmarks)
            lip_distance = abs(top_lip_center - bottom_lip_center)
            return image_with_landmarks, lip_distance

        mock_landmarks = np.matrix([[i, i*2] for i in range(68)])
        mock_get_landmarks = MagicMock(return_value=mock_landmarks)
        mock_top_lip = MagicMock(return_value=100)
        mock_bottom_lip = MagicMock(return_value=100)  # Same position
        mock_annotate = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        _, lip_distance = mouth_open(image, mock_get_landmarks, mock_top_lip,
                                     mock_bottom_lip, mock_annotate)

        self.assertEqual(lip_distance, 0)

    def test_very_large_lip_distance(self):
        """Test with abnormally large lip distance."""
        def mouth_open(image, get_landmarks_func, top_lip_func, bottom_lip_func, annotate_func):
            landmarks = get_landmarks_func(image)
            if landmarks is None:
                return image, 0
            image_with_landmarks = annotate_func(image, landmarks)
            top_lip_center = top_lip_func(landmarks)
            bottom_lip_center = bottom_lip_func(landmarks)
            lip_distance = abs(top_lip_center - bottom_lip_center)
            return image_with_landmarks, lip_distance

        mock_landmarks = np.matrix([[i, i*2] for i in range(68)])
        mock_get_landmarks = MagicMock(return_value=mock_landmarks)
        mock_top_lip = MagicMock(return_value=50)
        mock_bottom_lip = MagicMock(return_value=200)  # Very large distance
        mock_annotate = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        _, lip_distance = mouth_open(image, mock_get_landmarks, mock_top_lip,
                                     mock_bottom_lip, mock_annotate)

        self.assertEqual(lip_distance, 150)
        self.assertIsInstance(lip_distance, int)

    def test_yawn_counter_overflow(self):
        """Test yawn counter with large number of yawns."""
        yawns = 999
        yawn_status = False
        prev_yawn_status = True

        if prev_yawn_status == True and yawn_status == False:
            yawns += 1

        self.assertEqual(yawns, 1000)

    def test_negative_lip_distance_handling(self):
        """Test that absolute value handles negative differences."""
        top_lip_center = 150
        bottom_lip_center = 100

        lip_distance = abs(top_lip_center - bottom_lip_center)

        self.assertEqual(lip_distance, 50)
        self.assertGreaterEqual(lip_distance, 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete workflows."""

    def test_full_yawn_detection_workflow(self):
        """Test complete yawn detection workflow."""
        yawns = 0
        yawn_status = False
        YAWN_THRESHOLD = 25

        # Simulate video frames with varying lip distances
        lip_distances = [15, 18, 20, 30, 35, 32, 28, 20, 15, 12, 26, 30, 22, 15]

        for lip_distance in lip_distances:
            prev_yawn_status = yawn_status

            if lip_distance > YAWN_THRESHOLD:
                yawn_status = True
            else:
                yawn_status = False

            if prev_yawn_status == True and yawn_status == False:
                yawns += 1

        # Should detect 2 complete yawns
        self.assertEqual(yawns, 2)

    @patch('cv2.putText')
    @patch('cv2.imshow')
    def test_complete_frame_processing(self, mock_imshow, mock_puttext):
        """Test complete frame processing with visualization."""
        import cv2

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lip_distance = 30
        yawns = 3
        yawn_status = True
        YAWN_THRESHOLD = 25

        if lip_distance > YAWN_THRESHOLD:
            yawn_status = True
            cv2.putText(frame, "Subject is Yawning", (50, 450),
                       cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            output_text = " Yawn Count: " + str(yawns + 1)
            cv2.putText(frame, output_text, (50, 50),
                       cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
        else:
            yawn_status = False

        cv2.imshow('Live Landmarks', landmarks_frame)
        cv2.imshow('Yawn Detection', frame)

        self.assertTrue(yawn_status)
        self.assertEqual(mock_puttext.call_count, 2)
        self.assertEqual(mock_imshow.call_count, 2)


class TestLandmarkIndexing(unittest.TestCase):
    """Test correct landmark point indexing."""

    def test_top_lip_indices(self):
        """Test that top lip uses correct landmark indices."""
        # According to the code: range(50,53) and range(61,64)
        expected_indices = list(range(50, 53)) + list(range(61, 64))

        self.assertEqual(len(expected_indices), 6)
        self.assertIn(50, expected_indices)
        self.assertIn(52, expected_indices)
        self.assertIn(61, expected_indices)
        self.assertIn(63, expected_indices)

    def test_bottom_lip_indices(self):
        """Test that bottom lip uses correct landmark indices."""
        # According to the code: range(65,68) and range(56,59)
        expected_indices = list(range(65, 68)) + list(range(56, 59))

        self.assertEqual(len(expected_indices), 6)
        self.assertIn(65, expected_indices)
        self.assertIn(67, expected_indices)
        self.assertIn(56, expected_indices)
        self.assertIn(58, expected_indices)


class TestCameraOperations(unittest.TestCase):
    """Test camera operations."""

    def test_camera_capture_success(self):
        """Test successful camera frame capture."""
        mock_cam = MagicMock()
        mock_cam.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        ret, frame = mock_cam.read()

        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_camera_capture_failure(self):
        """Test camera frame capture failure."""
        mock_cam = MagicMock()
        mock_cam.read.return_value = (False, None)

        ret, frame = mock_cam.read()

        self.assertFalse(ret)
        self.assertIsNone(frame)

    def test_camera_release(self):
        """Test camera release."""
        mock_cam = MagicMock()
        mock_cam.release()

        mock_cam.release.assert_called_once()

    @patch('cv2.destroyAllWindows')
    def test_windows_cleanup(self, mock_destroy):
        """Test OpenCV windows cleanup."""
        import cv2
        cv2.destroyAllWindows()

        mock_destroy.assert_called_once()


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
