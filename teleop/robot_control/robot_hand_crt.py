# CRT Gripper Controller for teleoperation
# Communicates via Modbus RTU over RS485 (no DDS)

import numpy as np
import time
import threading
from multiprocessing import Process, Value, Lock

import os
import sys
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

# Lazy import to avoid errors when gripper not connected
Gripper = None

def _import_gripper():
    global Gripper
    if Gripper is None:
        gripper_path = os.path.join(parent2_dir, 'gripper_ctek')
        if gripper_path not in sys.path:
            sys.path.insert(0, gripper_path)
        from crt_gripper import Gripper as _Gripper
        Gripper = _Gripper


class CRT_Gripper_Controller:
    """
    Controller for CRT ctag2f90d grippers via Modbus RTU.

    Uses a simple threshold-based control:
    - Input value < OPEN_THRESHOLD: Open gripper
    - Input value > CLOSE_THRESHOLD: Close with predefined force
    - Otherwise: Move to proportional width
    """

    # Control thresholds for hand tracking mode (pinch value range 5.0-7.0)
    PINCH_OPEN_THRESHOLD = 5.5      # Below this: open gripper
    PINCH_CLOSE_THRESHOLD = 6.5     # Above this: close/grip

    # Control thresholds for controller mode (trigger value range 0.0-1.0)
    TRIGGER_OPEN_THRESHOLD = 0.3    # Below this: open gripper
    TRIGGER_CLOSE_THRESHOLD = 0.7   # Above this: close/grip

    # Gripper parameters
    DEFAULT_FORCE = 50        # Default grip force (1-100%)
    DEFAULT_SPEED = 50        # Default movement speed (1-100%)
    GRIP_SPEED = 30           # Speed when gripping (slower for safety)
    WIDTH_OPEN = 90.0         # Fully open width in mm
    WIDTH_CLOSED = 0.0        # Fully closed width in mm

    def __init__(self,
                 left_gripper_value_in,
                 right_gripper_value_in,
                 dual_gripper_data_lock=None,
                 dual_gripper_state_out=None,
                 dual_gripper_action_out=None,
                 left_port='/dev/ttyACM0',
                 right_port='/dev/ttyACM1',
                 fps=30.0,
                 simulation_mode=False):
        """
        Initialize CRT gripper controller.

        Args:
            left_gripper_value_in: [input] multiprocessing.Value for left gripper command (0.0-7.0)
            right_gripper_value_in: [input] multiprocessing.Value for right gripper command
            dual_gripper_data_lock: Lock for synchronizing state/action arrays
            dual_gripper_state_out: [output] multiprocessing.Array for gripper state (width in mm)
            dual_gripper_action_out: [output] multiprocessing.Array for gripper action (target width)
            left_port: Serial port for left gripper
            right_port: Serial port for right gripper
            fps: Control loop frequency
            simulation_mode: If True, skip hardware initialization
        """
        logger_mp.info("Initialize CRT_Gripper_Controller...")

        self.fps = fps
        self.simulation_mode = simulation_mode
        self.left_port = left_port
        self.right_port = right_port

        # State values for feedback (width in mm)
        self.left_state_value = Value('d', self.WIDTH_OPEN, lock=True)
        self.right_state_value = Value('d', self.WIDTH_OPEN, lock=True)

        # Track last commanded state to avoid redundant commands
        self.left_last_cmd = None
        self.right_last_cmd = None

        if not self.simulation_mode:
            _import_gripper()

            # Initialize grippers
            try:
                self.left_gripper = Gripper(port=left_port, auto_enable=True)
                logger_mp.info(f"Left CRT gripper connected on {left_port}")
            except Exception as e:
                logger_mp.error(f"Failed to connect left CRT gripper on {left_port}: {e}")
                self.left_gripper = None

            try:
                self.right_gripper = Gripper(port=right_port, auto_enable=True)
                logger_mp.info(f"Right CRT gripper connected on {right_port}")
            except Exception as e:
                logger_mp.error(f"Failed to connect right CRT gripper on {right_port}: {e}")
                self.right_gripper = None

            if self.left_gripper is None and self.right_gripper is None:
                raise RuntimeError("No CRT grippers connected. Check serial ports.")

            # Start feedback thread
            self.feedback_thread = threading.Thread(target=self._update_feedback_state)
            self.feedback_thread.daemon = True
            self.feedback_thread.start()

            # Wait for initial state
            time.sleep(0.2)
            logger_mp.info("[CRT_Gripper_Controller] Feedback thread started.")
        else:
            self.left_gripper = None
            self.right_gripper = None
            logger_mp.info("[CRT_Gripper_Controller] Simulation mode - no hardware.")

        # Start control thread
        self.gripper_control_thread = threading.Thread(
            target=self.control_thread,
            args=(left_gripper_value_in, right_gripper_value_in,
                  self.left_state_value, self.right_state_value,
                  dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out)
        )
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize CRT_Gripper_Controller OK!")

    def _update_feedback_state(self):
        """Thread that periodically reads gripper state."""
        while True:
            try:
                if self.left_gripper is not None:
                    left_width = self.left_gripper.get_width()
                    with self.left_state_value.get_lock():
                        self.left_state_value.value = left_width
            except Exception as e:
                logger_mp.debug(f"Failed to read left gripper state: {e}")

            try:
                if self.right_gripper is not None:
                    right_width = self.right_gripper.get_width()
                    with self.right_state_value.get_lock():
                        self.right_state_value.value = right_width
            except Exception as e:
                logger_mp.debug(f"Failed to read right gripper state: {e}")

            time.sleep(0.05)  # ~20Hz feedback rate

    def _compute_target(self, input_value):
        """
        Compute target width and command type from input value.

        Automatically detects input type based on value range:
        - Trigger values: 0.0-1.0 (controller mode)
        - Pinch values: 5.0-7.0 (hand tracking mode)

        Returns:
            tuple: (cmd_type, target_width)
        """
        # Detect input type based on value range
        if input_value <= 2.0:
            # Controller trigger mode (0.0-1.0 range)
            open_thresh = self.TRIGGER_OPEN_THRESHOLD
            close_thresh = self.TRIGGER_CLOSE_THRESHOLD
        else:
            # Hand tracking pinch mode (5.0-7.0 range)
            open_thresh = self.PINCH_OPEN_THRESHOLD
            close_thresh = self.PINCH_CLOSE_THRESHOLD

        if input_value < open_thresh:
            return 'open', self.WIDTH_OPEN
        elif input_value > close_thresh:
            return 'close', self.WIDTH_CLOSED
        else:
            # Proportional control between thresholds
            ratio = (input_value - open_thresh) / (close_thresh - open_thresh)
            target_width = self.WIDTH_OPEN - ratio * (self.WIDTH_OPEN - self.WIDTH_CLOSED)
            return 'move', target_width

    def _command_gripper(self, gripper, input_value, last_cmd):
        """
        Determine and send command to gripper based on input value.

        Returns:
            tuple: (new_last_cmd, target_width)
        """
        cmd, target_width = self._compute_target(input_value)

        # Only send command to hardware if gripper is connected and command changed
        if gripper is not None and cmd != last_cmd:
            try:
                if cmd == 'open':
                    gripper.open(speed=self.DEFAULT_SPEED, blocking=False)
                elif cmd == 'close':
                    gripper.close(force=self.DEFAULT_FORCE, speed=self.GRIP_SPEED, blocking=False)
                # For 'move', we could call move_to_width but it's noisy for small changes
            except Exception as e:
                logger_mp.warning(f"Failed to command gripper: {e}")

        return cmd, target_width

    def control_thread(self, left_gripper_value_in, right_gripper_value_in,
                       left_state_value, right_state_value,
                       dual_gripper_data_lock=None,
                       dual_gripper_state_out=None, dual_gripper_action_out=None):
        """Main control loop."""
        self.running = True

        try:
            while self.running:
                start_time = time.time()

                # Read input values from XR device
                with left_gripper_value_in.get_lock():
                    left_input = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_input = right_gripper_value_in.value

                # Get current state
                with left_state_value.get_lock():
                    left_state = left_state_value.value
                with right_state_value.get_lock():
                    right_state = right_state_value.value

                # Command grippers (only if input has been initialized)
                if left_input != 0.0 or right_input != 0.0:
                    if not self.simulation_mode:
                        self.left_last_cmd, left_target = self._command_gripper(
                            self.left_gripper, left_input, self.left_last_cmd)
                        self.right_last_cmd, right_target = self._command_gripper(
                            self.right_gripper, right_input, self.right_last_cmd)
                    else:
                        # Simulation: just compute target widths
                        _, left_target = self._command_gripper(None, left_input, None)
                        _, right_target = self._command_gripper(None, right_input, None)
                        # In sim, state follows action immediately
                        left_state = left_target
                        right_state = right_target
                else:
                    left_target = self.WIDTH_OPEN
                    right_target = self.WIDTH_OPEN

                # Update output arrays
                if dual_gripper_state_out is not None and dual_gripper_action_out is not None:
                    with dual_gripper_data_lock:
                        dual_gripper_state_out[0] = left_state
                        dual_gripper_state_out[1] = right_state
                        dual_gripper_action_out[0] = left_target
                        dual_gripper_action_out[1] = right_target

                # Maintain loop frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / self.fps) - elapsed)
                time.sleep(sleep_time)

        finally:
            # Cleanup: open grippers on exit for safety
            if not self.simulation_mode:
                try:
                    if self.left_gripper is not None:
                        self.left_gripper.open(blocking=False)
                except:
                    pass
                try:
                    if self.right_gripper is not None:
                        self.right_gripper.open(blocking=False)
                except:
                    pass
            logger_mp.info("CRT_Gripper_Controller has been closed.")

    def close(self):
        """Stop the controller and clean up."""
        self.running = False
        if not self.simulation_mode:
            if self.left_gripper is not None:
                try:
                    self.left_gripper.disable()
                    self.left_gripper.disconnect()
                except:
                    pass
            if self.right_gripper is not None:
                try:
                    self.right_gripper.disable()
                    self.right_gripper.disconnect()
                except:
                    pass


if __name__ == "__main__":
    # Simple test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--left-port', type=str, default='/dev/ttyACM0')
    parser.add_argument('--right-port', type=str, default='/dev/ttyACM1')
    parser.add_argument('--sim', action='store_true')
    args = parser.parse_args()

    # Create input/output values
    left_gripper_value = Value('d', 0.0, lock=True)
    right_gripper_value = Value('d', 0.0, lock=True)
    dual_gripper_data_lock = Lock()
    dual_gripper_state_array = Array('d', 2, lock=False)
    dual_gripper_action_array = Array('d', 2, lock=False)

    from multiprocessing import Array
    dual_gripper_state_array = Array('d', 2, lock=False)
    dual_gripper_action_array = Array('d', 2, lock=False)

    ctrl = CRT_Gripper_Controller(
        left_gripper_value, right_gripper_value,
        dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array,
        left_port=args.left_port, right_port=args.right_port,
        simulation_mode=args.sim
    )

    print("CRT Gripper Controller started. Testing...")
    print("Simulating pinch values: 5.0 (open), 6.0 (mid), 7.0 (close)")

    try:
        # Test open
        print("\n[Test 1] Setting pinch to 5.0 (should OPEN)")
        left_gripper_value.value = 5.0
        right_gripper_value.value = 5.0
        time.sleep(2)
        print(f"State: L={dual_gripper_state_array[0]:.1f}mm, R={dual_gripper_state_array[1]:.1f}mm")

        # Test close
        print("\n[Test 2] Setting pinch to 7.0 (should CLOSE)")
        left_gripper_value.value = 7.0
        right_gripper_value.value = 7.0
        time.sleep(2)
        print(f"State: L={dual_gripper_state_array[0]:.1f}mm, R={dual_gripper_state_array[1]:.1f}mm")

        # Test mid
        print("\n[Test 3] Setting pinch to 6.0 (should be ~45mm)")
        left_gripper_value.value = 6.0
        right_gripper_value.value = 6.0
        time.sleep(2)
        print(f"State: L={dual_gripper_state_array[0]:.1f}mm, R={dual_gripper_state_array[1]:.1f}mm")

        # Open again
        print("\n[Test 4] Back to 5.0 (OPEN)")
        left_gripper_value.value = 5.0
        right_gripper_value.value = 5.0
        time.sleep(2)
        print(f"State: L={dual_gripper_state_array[0]:.1f}mm, R={dual_gripper_state_array[1]:.1f}mm")

    except KeyboardInterrupt:
        pass
    finally:
        ctrl.close()
        print("\nTest complete.")
