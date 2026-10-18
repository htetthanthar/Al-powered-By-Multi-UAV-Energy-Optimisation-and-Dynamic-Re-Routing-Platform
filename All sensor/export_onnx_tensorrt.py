# export_onnx_tensorrt.py
import os
import torch
from pathlib import Path

# Import model architectures with fallbacks
try:
    from models.energy_mlp import EnergySurrogateMLP
except ModuleNotFoundError:
    from dynamic_planner import EnergySurrogateMLP

try:
    from models.weather_lstm import BidirectionalWeatherLSTM
except ModuleNotFoundError:
    from train_weather import BidirectionalWeatherLSTM


def export_energy_mlp(output_path: str = "checkpoints/onnx/energy_surrogate.onnx") -> str:
    """Exports the trained EnergySurrogateMLP PyTorch model to ONNX."""
    onnx_file = Path(output_path)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    mlp_checkpoint = Path("checkpoints/best_energy_mlp.pth")
    model = EnergySurrogateMLP(in_features=5, hidden_dim=64)

    if mlp_checkpoint.exists():
        model.load_state_dict(torch.load(mlp_checkpoint, map_location="cpu"))
        print(f"[*] Loaded weights from {mlp_checkpoint}")
    else:
        print("[!] Warning: MLP checkpoint not found. Exporting initialized architecture.")

    model.eval()
    dummy_input = torch.randn(1, 5, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_file),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["flight_features"],
        output_names=["predicted_power_w"],
        dynamic_axes={
            "flight_features": {0: "batch_size"},
            "predicted_power_w": {0: "batch_size"}
        }
    )
    print(f"[+] Successfully exported Energy Surrogate ONNX: {onnx_file.resolve()}")
    return str(onnx_file)


def export_weather_lstm(output_path: str = "checkpoints/onnx/weather_lstm.onnx") -> str:
    """Exports the trained BidirectionalWeatherLSTM PyTorch model to ONNX."""
    onnx_file = Path(output_path)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    lstm_checkpoint = Path("checkpoints/best_weather_lstm.pth")
    model = BidirectionalWeatherLSTM(in_features=4, hidden_dim=32, num_layers=2, out_steps=6)

    if lstm_checkpoint.exists():
        model.load_state_dict(torch.load(lstm_checkpoint, map_location="cpu"))
        print(f"[*] Loaded weights from {lstm_checkpoint}")
    else:
        print("[!] Warning: Weather LSTM checkpoint not found. Exporting initialized architecture.")

    model.eval()
    dummy_input = torch.randn(1, 12, 4, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_file),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["weather_sequence"],
        output_names=["forecast_horizon"],
        dynamic_axes={
            "weather_sequence": {0: "batch_size"},
            "forecast_horizon": {0: "batch_size"}
        }
    )
    print(f"[+] Successfully exported Weather LSTM ONNX: {onnx_file.resolve()}")
    return str(onnx_file)


def export_all_models_to_onnx():
    export_energy_mlp()
    export_weather_lstm()


if __name__ == "__main__":
    export_all_models_to_onnx()