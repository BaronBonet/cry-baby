import datetime
import json
import pathlib
from cry_baby.app.core.ports import Repository


class JSONRepo(Repository):
    def __init__(self, json_file_path: pathlib.Path):
        self.json_file_path = json_file_path
        if not self.json_file_path.exists():
            with open(self.json_file_path, "w") as file:
                json.dump([], file)

    def save(self, audio_file_path: pathlib.Path, prediction: float):
        with open(self.json_file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []

        timestamp = datetime.datetime.now().isoformat()
        data.append(
            {
                "timestamp": timestamp,
                "audio_file_path": str(audio_file_path),
                "prediction": float(prediction),
            }
        )

        with open(self.json_file_path, "w") as file:
            json.dump(data, file, indent=4)
