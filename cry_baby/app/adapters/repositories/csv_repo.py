import datetime
import pathlib
from cry_baby.app.core.ports import Repository


class CSVRepo(Repository):
    def __init__(self, csv_file_path: pathlib.Path):
        self.csv_file_path = csv_file_path

    def save(self, audio_file_path: pathlib.Path, prediction: float):
        with open(self.csv_file_path, "a") as file:
            match file.tell():
                case 0:
                    file.write("timestamp,audio_file_path,prediction\n")
                case _ if file.tell() > 0:
                    timestamp = datetime.datetime.now().isoformat()
                    file.write(f"{timestamp},{audio_file_path},{prediction}\n")
                case _:
                    raise ValueError("File pointer is negative")
