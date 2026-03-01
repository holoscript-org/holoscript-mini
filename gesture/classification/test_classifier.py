"""Test script for gesture classifier and debounce manager.

Demonstrates the classification logic with mock landmarks.
"""

import time
from gesture_classifier import GestureClassifier
from debounce_manager import DebounceManager


class MockLandmark:
    """Mock landmark for testing."""
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def create_open_palm():
    """Create landmarks for open palm gesture."""
    landmarks = []
    # Simplified: fingertips have lower y (above) their PIPs
    # Thumb is far from index (open palm)
    for i in range(21):
        if i == 4:  # Thumb tip - far left
            landmarks.append(MockLandmark(0.3, 0.4, 0.0))
        elif i == 8:  # Index tip - open, far from thumb
            landmarks.append(MockLandmark(0.6, 0.3, 0.0))
        elif i in [12, 16, 20]:  # Other tips - open
            landmarks.append(MockLandmark(0.5, 0.3, 0.0))
        elif i in [3, 6, 10, 14, 18]:  # PIPs/IP
            landmarks.append(MockLandmark(0.5, 0.5, 0.0))
        else:
            landmarks.append(MockLandmark(0.5, 0.6, 0.0))
    return landmarks


def create_fist():
    """Create landmarks for fist gesture."""
    landmarks = []
    # Fingertips have higher y (below) their PIPs = closed
    # Thumb is far from index
    for i in range(21):
        if i == 4:  # Thumb tip - closed but not near index
            landmarks.append(MockLandmark(0.4, 0.6, 0.0))
        elif i == 8:  # Index tip - closed
            landmarks.append(MockLandmark(0.6, 0.7, 0.0))
        elif i in [12, 16, 20]:  # Other tips closed
            landmarks.append(MockLandmark(0.5, 0.7, 0.0))
        elif i in [3, 6, 10, 14, 18]:  # PIPs/IP
            landmarks.append(MockLandmark(0.5, 0.5, 0.0))
        else:
            landmarks.append(MockLandmark(0.5, 0.4, 0.0))
    return landmarks


def create_pinch():
    """Create landmarks for pinch gesture."""
    landmarks = []
    for i in range(21):
        if i == 4:  # Thumb tip
            landmarks.append(MockLandmark(0.5, 0.4, 0.0))
        elif i == 8:  # Index tip - very close to thumb
            landmarks.append(MockLandmark(0.51, 0.4, 0.0))
        elif i in [12, 16, 20]:  # Other tips closed
            landmarks.append(MockLandmark(0.5, 0.7, 0.0))
        elif i in [3, 6, 10, 14, 18]:  # PIPs
            landmarks.append(MockLandmark(0.5, 0.5, 0.0))
        else:
            landmarks.append(MockLandmark(0.5, 0.6, 0.0))
    return landmarks


def create_point():
    """Create landmarks for pointing gesture."""
    landmarks = []
    for i in range(21):
        if i == 4:  # Thumb tip - closed, not near index
            landmarks.append(MockLandmark(0.4, 0.6, 0.0))
        elif i == 8:  # Index tip - open
            landmarks.append(MockLandmark(0.6, 0.3, 0.0))
        elif i in [12, 16, 20]:  # Other tips closed
            landmarks.append(MockLandmark(0.5, 0.7, 0.0))
        elif i in [3, 6, 10, 14, 18]:  # PIPs
            landmarks.append(MockLandmark(0.5, 0.5, 0.0))
        else:
            landmarks.append(MockLandmark(0.5, 0.6, 0.0))
    return landmarks


def create_v_sign():
    """Create landmarks for V sign gesture."""
    landmarks = []
    for i in range(21):
        if i == 4:  # Thumb tip - closed, not near index
            landmarks.append(MockLandmark(0.4, 0.6, 0.0))
        elif i in [8, 12]:  # Index and middle tips - open
            landmarks.append(MockLandmark(0.6, 0.3, 0.0))
        elif i in [16, 20]:  # Other tips closed
            landmarks.append(MockLandmark(0.5, 0.7, 0.0))
        elif i in [3, 6, 10, 14, 18]:  # PIPs
            landmarks.append(MockLandmark(0.5, 0.5, 0.0))
        else:
            landmarks.append(MockLandmark(0.5, 0.6, 0.0))
    return landmarks


def main():
    """Test gesture classifier and debounce manager."""
    print("=" * 60)
    print("Gesture Classifier Test")
    print("=" * 60)
    
    classifier = GestureClassifier()
    debounce = DebounceManager(debounce_time=0.5)
    
    # Test gestures
    test_cases = [
        ("OPEN_PALM", create_open_palm()),
        ("FIST", create_fist()),
        ("PINCH", create_pinch()),
        ("POINT", create_point()),
        ("V_SIGN", create_v_sign()),
    ]
    
    print("\n1. Testing Gesture Classification:")
    print("-" * 60)
    for expected, landmarks in test_cases:
        gesture, confidence = classifier.classify(landmarks)
        status = "✓" if gesture == expected else "✗"
        print(f"{status} Expected: {expected:12} | Got: {gesture:12} | Conf: {confidence:.1f}")
    
    print("\n2. Testing Debounce Manager:")
    print("-" * 60)
    
    # Test gesture change - should trigger
    print("First PINCH gesture:")
    print(f"  Should trigger: {debounce.should_trigger('PINCH')}")  # True
    
    # Test same gesture immediately - should NOT trigger
    print("Immediate repeat PINCH:")
    print(f"  Should trigger: {debounce.should_trigger('PINCH')}")  # False
    
    # Test different gesture - should trigger immediately
    print("Change to FIST:")
    print(f"  Should trigger: {debounce.should_trigger('FIST')}")  # True
    
    # Test UNKNOWN - should never trigger
    print("UNKNOWN gesture:")
    print(f"  Should trigger: {debounce.should_trigger('UNKNOWN')}")  # False
    
    # Test same gesture after delay
    print("Wait 0.6 seconds, then repeat FIST...")
    time.sleep(0.6)
    print(f"  Should trigger: {debounce.should_trigger('FIST')}")  # True
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
