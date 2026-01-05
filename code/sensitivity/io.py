import numpy as np
import pandas as pd

def save_results(result, path="sensitivity_results.npz"):
    np.savez(
        path,
        X=result["X"],
        Y=result["Y"],
        failures=result["failures"],
    )

def load_results(path="sensitivity_results.npz"):
    data = np.load(path)
    return {
        "X": data["X"],
        "Y": data["Y"],
        "failures": data["failures"],
    }

def to_dataframe(result):
    df = pd.DataFrame(
        result["X"],
        columns=["mu", "lambda_prop", "prob"]
    )
    df["HV"] = result["Y"]
    df["error"] = result["failures"]
    df["lambda"] = (df["mu"] * df["lambda_prop"]).astype(int)
    return df
