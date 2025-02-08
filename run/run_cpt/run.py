import os
import subprocess
import sys
import glob
from config.cfg_cpt import cfg_cpt

def kill_python_processes_windows(current_pid):
    """Kill all running Python processes in Windows, except this script."""
    print("Terminating Windows Python processes...")
    try:
        tasklist = subprocess.check_output("tasklist", shell=True, encoding='cp1252')
        pids = [line.split()[1] for line in tasklist.splitlines() if 'python' in line.lower() and line.split()[1] != str(current_pid)]
        for pid in pids:
            subprocess.run(["taskkill", "/F", "/PID", pid], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while killing processes: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

def kill_python_processes_unix(current_pid):
    """Kill all running Python processes in Unix-like systems, except this script."""
    print("Terminating Unix-like Python processes...")
    try:
        ps_output = subprocess.check_output("ps aux", shell=True).decode()
        pids = [line.split()[1] for line in ps_output.splitlines() if 'python' in line and 'grep' not in line and line.split()[1] != str(current_pid)]
        for pid in pids:
            subprocess.run(["kill", "-9", pid], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while killing processes: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

def remove_old_files():
    """Remove old result files."""
    print("Removing old results...")
    for ext in ['*.xlsx', '*.html', '*.prof', '*.json']:
        for file in glob.glob(ext):
            os.remove(file)
            print(f"Removed {file}")

def main():
    # Change to the directory where this script is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    current_pid = os.getpid()  # Store the current process ID

    # Kill Python processes
    if os.name == 'nt':  # Windows
        kill_python_processes_windows(current_pid)
    else:  # Unix-like
        kill_python_processes_unix(current_pid)

    # Remove old files
    remove_old_files()

    # Start main.py with viztracer
    print("Starting main.py...")
    try:
        if cfg_cpt.profile:
            subprocess.run(["viztracer", "--tracer_entries", "10000000", "main.py"], check=True)
        else:
            subprocess.run(["python", "main.py"], check=True)
        print("main.py finished")
        if cfg_cpt.profile:
            subprocess.run(["vizviewer", "result.json"], check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to start main.py")
        sys.exit(1)

if __name__ == "__main__":
    main()