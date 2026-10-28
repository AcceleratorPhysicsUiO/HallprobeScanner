import os
import sys

sys.path.append(
    '/mn/fys-server1/a1/pierredr/Documents/4-pythonLibraries/pyTHM1176'
)

import time
import threading
from datetime import datetime
from pathlib import Path
import traceback

import numpy as np
import matplotlib.pyplot as plt
import libximc.highlevel as ximc
import usbtmc as backend
import pyTHM1176.api.thm_usbtmc_api as thm_api


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

# Motor calibration
L_cal = 500e-3
steps_cal = 25000
L_step = L_cal / steps_cal  # Motor step size [m / step]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

axis = None


# ---------------------------------------------------------------------
# Motor functions
# ---------------------------------------------------------------------

def start_motor():
    """Connect to the motor controller."""
    global axis

    axis = ximc.Axis("xi-com:///dev/ttyACM0")
    axis.open_device()


def stop_motor():
    """Disconnect from the motor controller."""
    global axis

    if axis is not None:
        axis.close_device()
        axis = None


def mToStep(val_m):
    """Convert a position in metres to motor steps."""
    return int(round(val_m / L_step))


def stepTom(val_steps):
    """Convert a motor position in steps to metres."""
    return float(val_steps) * L_step


# ---------------------------------------------------------------------
# Hall probe functions
# ---------------------------------------------------------------------

def start_acquisition(thm):
    """Start Hall-probe acquisition in a background thread."""
    thm.stop = False

    acquisition_thread = threading.Thread(
        target=thm.start_acquisition,
        name="THM1176-acquisition"
    )

    acquisition_thread.start()

    return acquisition_thread


def stop_acquisition(thm, acquisition_thread, timeout_s=5):
    """Stop Hall-probe acquisition and wait for the thread to exit."""
    if thm is None:
        return True

    thm.stop = True

    if acquisition_thread is None:
        return True

    if not acquisition_thread.is_alive():
        return True

    print("\nStopping Hall probe acquisition...")
    acquisition_thread.join(timeout=timeout_s)

    if acquisition_thread.is_alive():
        print(
            f"Warning: acquisition thread did not stop within "
            f"{timeout_s} seconds."
        )
        return False

    print("Hall probe acquisition stopped.")
    return True


def get_acquired_data(thm, item_names):
    """Copy acquired data into NumPy arrays."""
    required_names = ['Timestamp'] + item_names
    arrays = {}

    for name in required_names:
        if name not in thm.data_stack:
            raise KeyError(
                f"Expected channel '{name}' was not found in "
                f"thm.data_stack."
            )

        arrays[name] = np.asarray(thm.data_stack[name])

    n_samples = min(len(values) for values in arrays.values())

    if n_samples == 0:
        raise RuntimeError("No Hall-probe data were acquired.")

    return {
        name: values[:n_samples].copy()
        for name, values in arrays.items()
    }


def select_data_at_target_positions(data, position_intervals):
    """
    Keep only Hall-probe samples measured while the stage was stationary
    at one of the requested target positions.
    """
    selected_indices = []
    selected_position_m = []
    selected_position_steps = []

    n_samples = len(data['Timestamp'])

    for interval in position_intervals:
        start_index = min(interval['start_index'], n_samples)
        end_index = min(interval['end_index'], n_samples)

        if end_index <= start_index:
            print(
                f"Warning: no Hall-probe samples recorded at "
                f"{interval['target_m'] * 1e3:.3f} mm."
            )
            continue

        indices = np.arange(start_index, end_index)

        selected_indices.append(indices)

        selected_position_m.append(
            np.full(
                len(indices),
                interval['target_m'],
                dtype=float
            )
        )

        selected_position_steps.append(
            np.full(
                len(indices),
                interval['target_steps'],
                dtype=int
            )
        )

    if not selected_indices:
        raise RuntimeError(
            "No Hall-probe samples were acquired at target positions."
        )

    selected_indices = np.concatenate(selected_indices)

    filtered_data = {
        key: np.asarray(values)[selected_indices]
        for key, values in data.items()
    }

    filtered_data['StagePosition_m'] = np.concatenate(
        selected_position_m
    )

    filtered_data['StagePosition_steps'] = np.concatenate(
        selected_position_steps
    )

    return filtered_data


def save_data_to_csv(data, output_file):
    """
    Save Hall-probe data and target stage position to a CSV file.

    Bx, By, and Bz are stored in T.
    """
    table = np.column_stack((
        -data['By'],
        -data['Bz'],
        -data['Bx'],
        data['Temperature'],
        data['StagePosition_m'],
        data['StagePosition_steps']
    ))

    np.savetxt(
        output_file,
        table,
        delimiter=",",
        header=(
            "Bx,"
            "By,"
            "Bz,"
            "Temperature,"
            "StagePosition_m,"
            "StagePosition_steps"
        ),
        comments="",
        fmt="%.12g"
    )

    print(f"\nData saved to:\n  {output_file}")


def plot_final_data(
    data,
    item_names,
    labels,
    curve_type,
    to_show,
    png_output_file
):
    """
    Plot Hall-probe data against stage position, save the plot as PNG,
    and display it.

    Only data measured at requested target positions are plotted.
    """
    stage_position_mm = data['StagePosition_m'] * 1e3

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Stage position [mm]")
    ax1.set_ylabel("Magnetic field [mT]")

    fig.suptitle(
        f"Hall probe measurement \n at {x}, {y} "
        f"using {range_B} range at {timestamp}"
    )

    field_colors = ['tab:blue', 'tab:green', 'tab:red']

    lines = []
    visible_labels = []
    field_count = 0

    for k, show_curve in enumerate(to_show):
        if not show_curve:
            continue

        values = -data[item_names[k]] #"-" sign to flip all fields according to convention

        if curve_type[k] == 'F':
            line, = ax1.plot(
                stage_position_mm,
                1e3 * values,
                marker='o',
                linestyle='None',
                markersize=3,
                color=field_colors[field_count],
                label=labels[k]
            )

            field_count += 1
            lines.append(line)
            visible_labels.append(labels[k])

    if lines:
        ax1.legend(lines, visible_labels, loc='best')

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save the final plot as a high-resolution PNG before displaying it.
    fig.savefig(
        png_output_file,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"Plot saved to:\n  {png_output_file}")

    plt.show()


def clear_acquisition_data(thm):
    """
    Clear Hall-probe data buffers.

    Call only when no acquisition thread is running.
    """
    if thm is None:
        return

    for key in list(thm.data_stack.keys()):
        values = thm.data_stack[key]

        if hasattr(values, "clear"):
            values.clear()
        else:
            thm.data_stack[key] = []

    plt.close('all')

    print("Hall-probe data cleared from memory.")


# ---------------------------------------------------------------------
# Live plot
# ---------------------------------------------------------------------

class LiveFieldPlot:
    """
    Display Bx, By, and Bz against stage position while the scan runs.

    Only samples measured during stationary dwell periods are added.
    """

    def __init__(self):
        plt.ion()

        self.fig, self.ax = plt.subplots()

        self.ax.set_xlabel("Stage position [mm]")
        self.ax.set_ylabel("Magnetic field [mT]")
        self.ax.set_title(
            f"Hall probe measurement\n"
            f"at {x}, {y} using {range_B} range at {timestamp}"
        )
        self.ax.grid(True, alpha=0.3)

        self.lines = {
            'By': self.ax.plot(
                [],
                [],
                marker='o',
                linestyle='None',
                markersize=3,
                color='tab:blue',
                label='Bx'
            )[0],
            'Bz': self.ax.plot(
                [],
                [],
                marker='o',
                linestyle='None',
                markersize=3,
                color='tab:green',
                label='By'
            )[0],
            'Bx': self.ax.plot(
                [],
                [],
                marker='o',
                linestyle='None',
                markersize=3,
                color='tab:red',
                label='Bz'
            )[0],
        }

        self.ax.legend(loc='best')

        self.position_mm = []

        self.field_mT = {
            'Bx': [],
            'By': [],
            'Bz': [],
        }

        self.current_target_mm = None
        self.last_index = 0

        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def number_of_complete_samples(thm):
        """
        Determine the last index available in every required channel.

        The acquisition thread may append values between reads, so using the
        shortest channel prevents accessing incomplete samples.
        """
        return min(
            len(thm.data_stack['Timestamp']),
            len(thm.data_stack['Bx']),
            len(thm.data_stack['By']),
            len(thm.data_stack['Bz'])
        )

    def begin_position(self, thm, target_m):
        """
        Begin live plotting for a stationary target position.

        All data acquired before this method is called are ignored. This
        excludes samples collected during stage motion.
        """
        self.current_target_mm = target_m * 1e3
        self.last_index = self.number_of_complete_samples(thm)

    def update(self, thm):
        """
        Read new Hall-probe samples and update the live figure.

        This method is intended to be called while the stage is stationary.
        """
        if self.current_target_mm is None:
            return

        n_samples = self.number_of_complete_samples(thm)

        if n_samples <= self.last_index:
            plt.pause(0.001)
            return

        start_index = self.last_index
        end_index = n_samples
        n_new_samples = end_index - start_index

        bx = np.asarray(
            -thm.data_stack['Bx'][start_index:end_index], # "-" sign to account for orientation convention
            dtype=float
        )

        by = np.asarray(
            -thm.data_stack['By'][start_index:end_index], # "-" sign to account for orientation convention
            dtype=float
        )

        bz = np.asarray(
            -thm.data_stack['Bz'][start_index:end_index], # "-" sign to account for orientation convention
            dtype=float
        )

        self.position_mm.extend(
            [self.current_target_mm] * n_new_samples
        )

        self.field_mT['Bx'].extend((bx * 1e3).tolist())
        self.field_mT['By'].extend((by * 1e3).tolist())
        self.field_mT['Bz'].extend((bz * 1e3).tolist())

        self.last_index = end_index

        x_values = np.asarray(self.position_mm)

        for field_name, line in self.lines.items():
            y_values = np.asarray(self.field_mT[field_name])
            line.set_data(x_values, y_values)

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Allows Matplotlib to process window events.
        plt.pause(0.001)

    def finalize(self):
        """Perform a final redraw and leave the live plot visible."""
        self.current_target_mm = None
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

if __name__ == '__main__':

    thm = None
    acquisition_thread = None
    data = None
    live_plot = None

    try:
        # -------------------------------------------------------------
        # Scan parameters
        # -------------------------------------------------------------
        z_init_m = 0e-3
        z_final_m = 250e-3
        dz_m = 2e-3

        range_B = '0.1T'
        range_B = '1T'

        x = 'm00690'  # String in microns
        y = 'p14650'  # String in microns

        duration_at_position = 1.0
        position_refresh_period = 0.05

        z_vec_m = np.arange(
            z_init_m,
            z_final_m + dz_m / 2,
            dz_m
        )

        # -------------------------------------------------------------
        # Hall-probe settings
        # -------------------------------------------------------------
        params = {
            'block_size': 5,
            'period': 1.0 / 20.0,
            'range': range_B,
            'average': 400,
            'format': 'ASCII'
        }

        #item_name = ['Bx', 'By', 'Bz', 'Temperature']
        #labels = ['hello', 'By_probe', 'Bz_probe', 'T']
        item_name = ['By', 'Bz', 'Bx', 'Temperature']
        labels = ['Bx', 'By', 'Bz', 'T']
        curve_type = ['F', 'F', 'F', 'T']

        to_show = [True, True, True, False]

        # -------------------------------------------------------------
        # Start motor
        # -------------------------------------------------------------
        start_motor()

        print(f"Motor step size: {L_step * 1e6:.3f} um")

        pos = axis.get_position()

        print(
            f"Initial position: {pos.Position} steps, "
            f"{pos.uPosition} microsteps"
        )
        print(
            f"Initial position: {stepTom(pos.Position) * 1e3:.3f} mm"
        )

        # -------------------------------------------------------------
        # Find and connect to Hall probe
        # -------------------------------------------------------------
        devices = backend.list_devices()

        if len(devices) == 0:
            raise RuntimeError("No THM1176 Hall probe was found.")

        thm = thm_api.Thm1176(devices[0], **params)

        device_id = thm.get_id()

        print("\nHall probe:")
        for key in thm.id_fields:
            print(f"{key}: {device_id[key]}")

        # -------------------------------------------------------------
        # Clear data before beginning a fresh scan
        # -------------------------------------------------------------
        clear_acquisition_data(thm)

        # -------------------------------------------------------------
        # Start Hall-probe acquisition and live plot
        # -------------------------------------------------------------
        acquisition_thread = start_acquisition(thm)

        live_plot = LiveFieldPlot()

        print("\nWaiting for Hall probe acquisition to start...")
        time.sleep(2)

        # Stores data-stack index ranges acquired while stationary.
        position_intervals = []

        # -------------------------------------------------------------
        # Scan stage positions
        # -------------------------------------------------------------
        for z_m in z_vec_m:
            target_steps = mToStep(z_m)

            print("\n--------------------------------------------")
            print(f"New target: {z_m * 1e3:.3f} mm")
            print(f"Target steps: {target_steps}")

            axis.command_move(target_steps, 0)

            print("Stage is moving...")

            while True:
                pos = axis.get_position()

                current_steps = pos.Position
                current_m = stepTom(current_steps)

                print(
                    f"\rPosition: {current_m * 1e3:.3f} mm "
                    f"({current_steps} steps)",
                    end="",
                    flush=True
                )

                if current_steps == target_steps:
                    break

                # Keep the GUI responsive while the motor moves.
                plt.pause(0.001)
                time.sleep(position_refresh_period)

            print("\nTarget position reached.")

            print(
                f"Recording for {duration_at_position:.1f} seconds "
                f"at {z_m * 1e3:.3f} mm..."
            )

            # Exclude all samples acquired before the stationary dwell.
            recording_start_index = len(thm.data_stack['Timestamp'])

            # Exclude motion samples from the live plot as well.
            live_plot.begin_position(thm, z_m)

            measurement_start = time.time()

            while time.time() - measurement_start < duration_at_position:
                pos = axis.get_position()

                current_steps = pos.Position
                current_m = stepTom(current_steps)

                elapsed = time.time() - measurement_start

                print(
                    f"\rAt {current_m * 1e3:.3f} mm — "
                    f"measurement time: {elapsed:.1f} / "
                    f"{duration_at_position:.1f} s",
                    end="",
                    flush=True
                )

                # Add samples received since the preceding update.
                live_plot.update(thm)

                time.sleep(position_refresh_period)

            # Include samples that arrived during the final refresh period.
            live_plot.update(thm)

            # Data from start to end belong to this target position.
            recording_end_index = len(thm.data_stack['Timestamp'])

            position_intervals.append({
                'start_index': recording_start_index,
                'end_index': recording_end_index,
                'target_m': z_m,
                'target_steps': target_steps
            })

            print("\nMeasurement finished at this position.")

        # -------------------------------------------------------------
        # Stop acquisition before accessing stored data
        # -------------------------------------------------------------
        thread_stopped = stop_acquisition(
            thm,
            acquisition_thread
        )

        if not thread_stopped:
            raise RuntimeError(
                "Acquisition thread is still running. "
                "Data cannot safely be saved."
            )

        if live_plot is not None:
            live_plot.update(thm)
            live_plot.finalize()

        # -------------------------------------------------------------
        # Keep only data acquired at requested target positions
        # -------------------------------------------------------------
        all_data = get_acquired_data(thm, item_name)

        data = select_data_at_target_positions(
            all_data,
            position_intervals
        )

        del all_data

        # -------------------------------------------------------------
        # Create output directories and filenames
        # -------------------------------------------------------------
        output_dir = Path(
            "/mn/fys-server1/a1/pierredr/Documents/3-code/"
            "HallprobeScanner/data/20261015"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        png_output_dir = output_dir / "png"
        png_output_dir.mkdir(parents=True, exist_ok=True)

        csv_output_dir = output_dir / "csv"
        csv_output_dir.mkdir(parents=True, exist_ok=True)

        range_B_csv = '1T'

        if range_B == '0.1T':
            range_B_csv = '100mT'

        output_file = csv_output_dir / (
            f"hall_probe_scan_x_{x}_y_{y}_{range_B_csv}_{timestamp}.csv"
        )

        png_output_file = png_output_dir / (
            f"hall_probe_scan_x_{x}_y_{y}_{range_B_csv}_{timestamp}.png"
        )

        # -------------------------------------------------------------
        # Save CSV
        # -------------------------------------------------------------
        save_data_to_csv(data, output_file)

        # -------------------------------------------------------------
        # Print final stage position
        # -------------------------------------------------------------
        pos = axis.get_position()

        print("\n--------------------------------------------")
        print(
            f"Final position: {pos.Position} steps, "
            f"{pos.uPosition} microsteps"
        )
        print(
            f"Final position: {stepTom(pos.Position) * 1e3:.3f} mm"
        )

        # -------------------------------------------------------------
        # Static final plot and PNG export
        # -------------------------------------------------------------
        plt.ioff()
        
        plot_final_data(
            data,
            item_name,
            labels,
            curve_type,
            to_show,
            png_output_file
        )
        
        # plot_final_data returns only after the displayed plot is closed.
        plt.close('all')
        
        # -------------------------------------------------------------
        # Return stage to zero
        # -------------------------------------------------------------
        home_steps = 0
        
        print("\n--------------------------------------------")
        print("Returning stage to zero...")
        
        axis.command_move(home_steps, 0)
        
        while True:
            pos = axis.get_position()
            current_steps = pos.Position
            current_m = stepTom(current_steps)
        
            print(
                f"\rPosition: {current_m * 1e3:.3f} mm "
                f"({current_steps} steps)",
                end="",
                flush=True
            )
        
            if current_steps == home_steps:
                break
        
            time.sleep(position_refresh_period)
        
        print("\nStage has returned to zero.")
        
        del data
        data = None


    except KeyboardInterrupt:
        print("\n\nExperiment stopped by user.")

    except Exception as error:
        print(f"\n\nError: {error}")
        traceback.print_exc()

    finally:
        stop_acquisition(thm, acquisition_thread)
        stop_motor()
