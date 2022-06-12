from pathlib import Path
import numpy as np


path_name = "models"


class ModelWeightsSaver:
    @staticmethod
    def save(model, name):
        save_name = name if name else model.name
        if save_name is None:
            print("Neither the model nor the save function have been provided with a name. Generating name")
            save_name = str(model).split()[-1]
        full_path_name = f"{path_name}/{save_name}"
        Path(full_path_name).mkdir(parents=True, exist_ok=True)
        i = 0
        for layer in model.layers:
            if layer.has_parameters:
                weights, bias = layer.get_parameters()
                if weights is not None:
                    np.save(f"{full_path_name}/weights_layer_{i}.npy", weights)
                if bias is not None:
                    np.save(f"{full_path_name}/bias_layer_{i}.npy", bias)
                i += 1
        print(f"Model {save_name} saved")


class ModelWeightsLoader:
    @staticmethod
    def load(model, name):
        """
        Careful! This method will load even if the layers of the created model are different from the ones loaded
        If you get a dimension mismatch after loading it's probably because of this.
        """
        load_name = name if name else model.name
        if load_name is None:
            raise ValueError("Neither the model nor the load function have been provided with a name. Cannot load")
        i = 0
        full_path_name = f"{path_name}/{load_name}"
        for layer in model.layers:
            if layer.has_parameters:
                weights = np.load(f"{full_path_name}/weights_layer_{i}.npy")
                try:
                    bias = np.load(f"{full_path_name}/bias_layer_{i}.npy")
                except FileNotFoundError:
                    bias = None
                layer.set_parameters(weights, bias)
                i += 1
        print(f"Model {load_name} loaded")
