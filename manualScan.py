import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

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
# Do not bring Matplotlib windows in front of the terminal.
plt.rcParams["figure.raise_window"] = False

import libximc.highlevel as ximc
import usbtmc as backend
import pyTHM1176.api.thm_usbtmc_api as thm_api

# -------------------------------------------------------------
# User-defined scan parameters
# -------------------------------------------------------------

# Fixed motorized Z position.
z_scan_m = 162.5e-3

# Choose which manual transverse coordinate is scanned.
manual_axis_name = "Y"   # "X" or "Y"

x_min_um = -1380      # [um] -1380 um means 6mm X gauge 
x_max_um = +2120      # [um]
x_stepsize_um = 100   # [um] 5 per rotation

y_min_um = -10000     # [um] -10 000 um means 24.650mm gauge-Y
y_max_um = +15000     # [um]
y_stepsize_um = 500   # [um] one full rotation

# 20 points at 40 Hz = approximately 0.5 s.
n_samples_per_range = 20
acquisition_frequency_hz = 40.0

# Where to store the data
output_dir = Path(
    "/mn/fys-server1/a1/pierredr/Documents/3-code/HallprobeScanner/data/20261019"
)
 
# -------------------------------------------------------------
# Create output directories and filenames
# -------------------------------------------------------------

output_dir.mkdir(parents=True, exist_ok=True)

png_output_dir = output_dir / "png"
png_output_dir.mkdir(parents=True, exist_ok=True)

csv_output_dir = output_dir / "csv"
csv_output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Some handling of the user-defined axis and scan ranges
# -------------------------------------------------------------

manual_axis_name = manual_axis_name.upper()

if manual_axis_name not in ("X", "Y"):
    raise ValueError("manual_axis_name must be either 'X' or 'Y'.")

# The coordinate not being scanned is fixed at zero and we initialise the range
if manual_axis_name == "X":
    fixed_axis_name = "Y"
    fixed_axis_position_m = 0.0
    
    manual_positions_um = np.arange(
        x_min_um,
        x_max_um + x_stepsize_um,
        x_stepsize_um,
        dtype=float
    )
    
else:
    fixed_axis_name = "X"
    fixed_axis_position_m = 0.0
    
    manual_positions_um = np.arange(
        y_min_um,
        y_max_um + y_stepsize_um,
        y_stepsize_um,
        dtype=float
    )    

# Same values in metres, stored in the CSV.
manual_positions_m = manual_positions_um * 1e-6

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
position_refresh_period = 0.05


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


def acquire_n_samples(
    thm,
    n_samples,
    manual_axis_position_m,
    fixed_axis_position_m,
    z_position_m
):

    """
    Acquire exactly n_samples from the Hall probe.

    Field convention used in the setup:
        Bx_user = -By_probe
        By_user = -Bz_probe
        Bz_user =  Bx_probe
    """
    required_channels = [
        'Timestamp',
        'Bx',
        'By',
        'Bz',
        'Temperature'
    ]

    def complete_sample_count():
        return min(
            len(thm.data_stack[channel])
            for channel in required_channels
        )

    start_index = complete_sample_count()

    print(f"Acquiring {n_samples} points...")

    while complete_sample_count() < start_index + n_samples:
        current_samples = complete_sample_count() - start_index

        print(
            f"\rSamples: {current_samples} / {n_samples}",
            end="",
            flush=True
        )

        # Keeps a possible Matplotlib window responsive.
        plt.pause(0.001)
        time.sleep(0.01)

    print()

    end_index = start_index + n_samples

    bx_probe = np.asarray(
        thm.data_stack['Bx'][start_index:end_index],
        dtype=float
    )

    by_probe = np.asarray(
        thm.data_stack['By'][start_index:end_index],
        dtype=float
    )

    bz_probe = np.asarray(
        thm.data_stack['Bz'][start_index:end_index],
        dtype=float
    )

    temperature = np.asarray(
        thm.data_stack['Temperature'][start_index:end_index],
        dtype=float
    )

    timestamps = np.asarray(
        thm.data_stack['Timestamp'][start_index:end_index]
    )

    return {
        'Timestamp': timestamps,

        # Conversion from probe coordinates to setup coordinates:
        'Bx': -by_probe,
        'By': -bz_probe,
        'Bz': bx_probe,

        'Temperature': temperature,

        # Manual transverse coordinate:
        'ManualPosition_m': np.full(
            n_samples,
            manual_axis_position_m,
            dtype=float
        ),

        # Fixed transverse coordinate:        
        'FixedPosition_m': np.full(
            n_samples,
            fixed_axis_position_m,
            dtype=float
            ),

        # Fixed motorized Z coordinate:
        'ZPosition_m': np.full(
            n_samples,
            z_position_m,
            dtype=float
        )
    }

def concatenate_data(data_list):
    """Concatenate data dictionaries from all transverse positions."""
    if not data_list:
        raise RuntimeError("No data were acquired.")

    return {
        key: np.concatenate([data[key] for data in data_list])
        for key in data_list[0]
    }

def save_data_to_csv(
    data,
    output_file,
    manual_axis_name,
    fixed_axis_name
):
    """
    Save Hall-probe data in the setup coordinate convention.

    Magnetic-field columns are always:
        Bx_T, By_T, Bz_T

    Position columns are always:
        X_position_m, Y_position_m, Z_position_m
    """
    if manual_axis_name == "X":
        x_position_m = data['ManualPosition_m']
        y_position_m = data['FixedPosition_m']

    elif manual_axis_name == "Y":
        x_position_m = data['FixedPosition_m']
        y_position_m = data['ManualPosition_m']

    else:
        raise ValueError(
            "manual_axis_name must be 'X' or 'Y'."
        )

    table = np.column_stack((
        data['Bx'],
        data['By'],
        data['Bz'],
        data['Temperature'],
        x_position_m,
        y_position_m,
        data['ZPosition_m']
    ))

    np.savetxt(
        output_file,
        table,
        delimiter=",",
        header=(
            "Bx_T,"
            "By_T,"
            "Bz_T,"
            "Temperature,"
            "X_position_m,"
            "Y_position_m,"
            "Z_position_m"
        ),
        comments="",
        fmt="%.12g"
    )

    print(f"\nData saved to:\n  {output_file}")

def plot_manual_transverse_scan(
    data,
    manual_axis_name,
    fixed_axis_name,
    fixed_axis_position_m,
    z_scan_m,
    range_B,
    png_output_file
):
    """
    Create the final plot for one Hall-probe range.

    The figure is saved but not displayed here. The caller should call
    plt.show() once after both final figures have been created.
    """
    position_um = data['ManualPosition_m'] * 1e6

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.set_xlabel(f"Manual {manual_axis_name} position [$\\mu$m]")
    ax.set_ylabel("Magnetic field [mT]")

    ax.set_title(
        f"Manual transverse Hall-probe scan\n"
        f"{fixed_axis_name} = {fixed_axis_position_m * 1e3:.3f} mm, "
        f"Z = {z_scan_m * 1e3:.3f} mm, "
        f"range = {range_B}\n"
        f"{timestamp}"
    )

    fields = {
        'Bx': ('tab:blue', 'Bx'),
        'By': ('tab:green', 'By'),
        'Bz': ('tab:red', 'Bz')
    }

    # Individual raw samples.
    for field_name, (color, label) in fields.items():
        ax.plot(
            position_um,
            data[field_name] * 1e3,
            marker='o',
            linestyle='None',
            markersize=3,
            alpha=0.30,
            color=color,
            label=f"{label}: individual points"
        )

    # Mean and standard deviation at every manual position.
    unique_positions_m = np.unique(data['ManualPosition_m'])

    for field_name, (color, label) in fields.items():
        means = []
        stds = []

        for position_m in unique_positions_m:
            mask = data['ManualPosition_m'] == position_m

            means.append(np.mean(data[field_name][mask]))
            stds.append(np.std(data[field_name][mask]))

        ax.errorbar(
            unique_positions_m * 1e6,
            np.asarray(means) * 1e3,
            yerr=np.asarray(stds) * 1e3,
            marker='o',
            linestyle='-',
            linewidth=1.5,
            capsize=3,
            color=color,
            label=f"{label}: mean ± std"
        )

    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()

    fig.savefig(
        png_output_file,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"Plot saved to:\n  {png_output_file}")

    return fig

class LiveManualScanPlot:
    """
    Live plot for one Hall-probe range.

    One point is added after each completed manual transverse position.
    The point is the mean of the 20 acquired samples, with error bars
    corresponding to one standard deviation.
    """

    def __init__(
        self,
        manual_axis_name,
        fixed_axis_name,
        fixed_axis_position_m,
        z_scan_m,
        range_B
    ):
        plt.ion()

        self.manual_axis_name = manual_axis_name
        self.fixed_axis_name = fixed_axis_name
        self.fixed_axis_position_m = fixed_axis_position_m
        self.z_scan_m = z_scan_m
        self.range_B = range_B

        self.fig, self.ax = plt.subplots(figsize=(9, 6))

        self.ax.set_xlabel(
            f"Manual {manual_axis_name} position [$\\mu$m]"
        )
        self.ax.set_ylabel("Magnetic field [mT]")

        self.ax.set_title(
            f"Live manual transverse Hall-probe scan\n"
            f"{fixed_axis_name} = {fixed_axis_position_m * 1e3:.3f} mm, "
            f"Z = {z_scan_m * 1e3:.3f} mm, "
            f"range = {range_B}"
        )

        self.ax.grid(True, alpha=0.3)

        self.colors = {
            'Bx': 'tab:blue',
            'By': 'tab:green',
            'Bz': 'tab:red'
        }

        self.positions_um = []

        self.means_mT = {
            'Bx': [],
            'By': [],
            'Bz': []
        }

        self.stds_mT = {
            'Bx': [],
            'By': [],
            'Bz': []
        }

        self.fig.tight_layout()
        self.fig.show()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_position_measurement(self, data_position):
        """
        Add the mean and standard deviation of one completed position.
        """
        position_um = data_position['ManualPosition_m'][0] * 1e6

        self.positions_um.append(position_um)

        for field_name in ['Bx', 'By', 'Bz']:
            values_mT = data_position[field_name] * 1e3

            self.means_mT[field_name].append(np.mean(values_mT))
            self.stds_mT[field_name].append(np.std(values_mT))

        # Clear and redraw: simple and robust for a small number of points.
        self.ax.clear()

        self.ax.set_xlabel(
            f"Manual {self.manual_axis_name} position [$\\mu$m]"
        )
        self.ax.set_ylabel("Magnetic field [mT]")

        self.ax.set_title(
            f"Live manual transverse Hall-probe scan\n"
            f"{self.fixed_axis_name} = "
            f"{self.fixed_axis_position_m * 1e3:.3f} mm, "
            f"Z = {self.z_scan_m * 1e3:.3f} mm, "
            f"range = {self.range_B}"
        )

        self.ax.grid(True, alpha=0.3)

        for field_name in ['Bx', 'By', 'Bz']:
            self.ax.errorbar(
                self.positions_um,
                self.means_mT[field_name],
                yerr=self.stds_mT[field_name],
                marker='o',
                linestyle='-',
                linewidth=1.5,
                capsize=3,
                color=self.colors[field_name],
                label=f"{field_name}: mean ± std"
            )

        self.ax.legend(loc='best')

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Allows the window to refresh and remain responsive.
        plt.pause(0.001)

    def finalize(self):
        """Perform a final live-plot refresh."""
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ---------------------------------------------------------------------
# Hall probe functions
# ---------------------------------------------------------------------
   
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

    print("Hall-probe data cleared from memory.")    

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
    
# ---------------------------------------------------------------------
# Moving the stage in Z
# ---------------------------------------------------------------------
    
start_motor()

print(f"Motor step size: {L_step * 1e6:.3f} um")

z_target_steps = mToStep(z_scan_m)

print("\n--------------------------------------------")
print(f"Moving Z stage to {z_scan_m * 1e3:.3f} mm...")
print(f"Target: {z_target_steps} steps")

axis.command_move(z_target_steps, 0)

while True:
    pos = axis.get_position()

    current_steps = pos.Position
    current_m = stepTom(current_steps)

    print(
        f"\rZ position: {current_m * 1e3:.3f} mm "
        f"({current_steps} steps)",
        end="",
        flush=True
    )

    if current_steps == z_target_steps:
        break

    time.sleep(position_refresh_period)

print("\nZ target position reached.")

# -------------------------------------------------------------
# Manual transverse scan
# -------------------------------------------------------------

data_by_range = {
    '0.1T': [],
    '1T': []
}

live_plots = {
    '0.1T': LiveManualScanPlot(
        manual_axis_name=manual_axis_name,
        fixed_axis_name=fixed_axis_name,
        fixed_axis_position_m=fixed_axis_position_m,
        z_scan_m=z_scan_m,
        range_B='0.1T'
    ),
    '1T': LiveManualScanPlot(
        manual_axis_name=manual_axis_name,
        fixed_axis_name=fixed_axis_name,
        fixed_axis_position_m=fixed_axis_position_m,
        z_scan_m=z_scan_m,
        range_B='1T'
    )
}

for manual_position_m in manual_positions_m:

    manual_position_um = manual_position_m * 1e6

    print("\n============================================")
    print(
        f"Please manually set the transverse stage to:\n"
        f"    {manual_axis_name} = {manual_position_um:.1f} um\n"
        f"while keeping:\n"
        f"    Z = {z_scan_m * 1e3:.3f} mm"
    )

    input(
        "\nWhen the manual stage is at the requested position, "
        "press ENTER to start the measurements..."
    )

    # Measure both Hall-probe ranges at the same manually validated
    # transverse coordinate.
    for range_B in ['0.1T', '1T']:

        print("\n--------------------------------------------")
        print(
            f"Measuring at {manual_axis_name} = "
            f"{manual_position_um:.1f} um"
        )
        print(f"Hall-probe range: {range_B}")

        params = {
            'block_size': 5,
            'period': 1.0 / acquisition_frequency_hz,
            'range': range_B,
            'average': 400,
            'format': 'ASCII'
        }

        # Find and connect to Hall probe
        devices = backend.list_devices()
        if len(devices) == 0:
            raise RuntimeError("No THM1176 Hall probe was found.")

        # Create a new Hall-probe instance for the selected range.
        thm = thm_api.Thm1176(devices[0], **params)

        clear_acquisition_data(thm)

        acquisition_thread = start_acquisition(thm)

        print("Waiting for Hall-probe acquisition to start...")
        time.sleep(0.2)

        data_position = acquire_n_samples(
            thm=thm,
            n_samples=n_samples_per_range,
            manual_axis_position_m=manual_position_m,
            fixed_axis_position_m=fixed_axis_position_m,
            z_position_m=z_scan_m
            )

        thread_stopped = stop_acquisition(
            thm,
            acquisition_thread
        )

        if not thread_stopped:
            raise RuntimeError(
                "Hall-probe acquisition thread did not stop."
            )

        acquisition_thread = None

        data_by_range[range_B].append(data_position)
        live_plots[range_B].add_position_measurement(data_position)

        print(
            f"Finished {range_B} measurement at "
            f"{manual_axis_name} = {manual_position_um:.1f} um."
        )

# -------------------------------------------------------------
# Save one CSV and one PNG for each Hall-probe range
# -------------------------------------------------------------
final_figures = []

for range_B in ['0.1T', '1T']:

    data = concatenate_data(data_by_range[range_B])

    range_filename = '100mT' if range_B == '0.1T' else '1T'

    csv_output_file = csv_output_dir / (
        f"hall_probe_manual_{manual_axis_name}_scan_"
        f"{fixed_axis_name}_{fixed_axis_position_m * 1e3:.3f}mm_"
        f"Z_{z_scan_m * 1e3:.3f}mm_"
        f"{range_filename}_"
        f"{timestamp}.csv"
    )

    png_output_file = png_output_dir / (
        f"hall_probe_manual_{manual_axis_name}_scan_"
        f"{fixed_axis_name}_{fixed_axis_position_m * 1e3:.3f}mm_"
        f"Z_{z_scan_m * 1e3:.3f}mm_"
        f"{range_filename}_"
        f"{timestamp}.png"
    )

    save_data_to_csv(
        data=data,
        output_file=csv_output_file,
        manual_axis_name=manual_axis_name,
        fixed_axis_name=fixed_axis_name
    )

    fig = plot_manual_transverse_scan(
        data=data,
        manual_axis_name=manual_axis_name,
        fixed_axis_name=fixed_axis_name,
        fixed_axis_position_m=fixed_axis_position_m,
        z_scan_m=z_scan_m,
        range_B=range_B,
        png_output_file=png_output_file
    )

    final_figures.append(fig)

for live_plot in live_plots.values():
    live_plot.finalize()

# Disable interactive mode, then display all figures together.
plt.ioff()

print("\nBoth final plots are ready. Close their windows to end the script.")
plt.show()
