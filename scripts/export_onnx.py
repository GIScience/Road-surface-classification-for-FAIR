#vendored from fairs pipeline integration fo the model
import tempfile
import zipfile
from pathlib import Path

import onnx
import torch
from torch import nn

from road_surface_classifier.model import load_model

WEIGHTS = Path(__file__).resolve().parents[1] / "data" / "weights"
HEAD = WEIGHTS / "head.pt"
ENCODER_ZIP = WEIGHTS / "dinov3_encoder.zip"
LICENSE = WEIGHTS / "LICENSE-DINOv3.md"
OUT = WEIGHTS / "dinov3s_road_surface.onnx"
PATCH_SIZE_PX = 256
MAX_PATCHES_PER_ROAD = 4


class MainOnly(nn.Module):
    """All patches of x form one bag; the attention weights are dropped."""

    def __init__(self, wrapped: nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrapped(x, [x.shape[0]])[0]


def main() -> None:
    with zipfile.ZipFile(ENCODER_ZIP) as archive:
        encoder = Path(archive.extract("encoder.pt", tempfile.mkdtemp()))
    model = MainOnly(load_model(HEAD, encoder, trainable="classifier").cpu().eval()).eval()
    dummy = torch.rand(MAX_PATCHES_PER_ROAD, 3, PATCH_SIZE_PX, PATCH_SIZE_PX)
    torch.onnx.export(
        model,
        (dummy,),
        str(OUT),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("patches", min=1, max=MAX_PATCHES_PER_ROAD)}},
        dynamo=True,
    )
    onnx_model = onnx.load(OUT)
    onnx.helper.set_model_props(onnx_model, {"license": LICENSE.read_text(encoding="utf-8"), "notice": "Built with DINOv3"})
    onnx.save(onnx_model, OUT, save_as_external_data=False)
    onnx.checker.check_model(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
