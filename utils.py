import json
import pandas as pd


def save_metrics(scores, output_file_path):
    with open(output_file_path, 'w+') as f:
        json.dump(scores, f, ensure_ascii=False)


def save_predictions(predictions, output_file_path):

    if output_file_path.endswith(".json"):
        with open(output_file_path, 'w+') as f:
            json.dump(predictions, f, ensure_ascii=False)
        print(f"[Predictions saved to {output_file_path}]")
        return
    elif output_file_path.endswith(".csv"):
        output_df = pd.DataFrame(predictions)
        output_df.to_csv(output_file_path, index=False)
        print(f"[Predictions saved to {output_file_path}]")
    else:
        raise ValueError("Invalid file format. Please use .json or .csv")