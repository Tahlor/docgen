import subprocess
import platform

import os
import subprocess
import platform


def map_drive(target_path: str, drive_letter: str) -> None:
    """
    Map a folder to a drive letter using the `subst` command.

    Args:
    - target_path (str): The folder path you want to map.
    - drive_letter (str): The drive letter you want to assign (e.g., "Z:").

    Returns:
    None
    """
    if platform.system() != "Windows":
        raise EnvironmentError("This function is designed for Windows only.")

    # Check if drive is already mapped
    if os.path.exists(drive_letter):
        print(f"The drive {drive_letter} is already mapped.")
        return

    command = f"subst {drive_letter} {target_path}"
    subprocess.run(command, check=True, shell=True)


# Example usage:
# map_drive("G:\\s3\\synthetic_data\\resources\\backgrounds\\synthetic_backgrounds\\dalle", "B:")


if __name__ == "__main__":
    target_path = r"G:\s3\synthetic_data\resources\backgrounds\synthetic_backgrounds\dalle"
    drive_letter = "Z:"
    map_drive(target_path, drive_letter)
