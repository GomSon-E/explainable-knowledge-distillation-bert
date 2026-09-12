import torch

from explainable_kd.common.runtime import describe_device, resolve_device


def test_auto_device_reports_the_selected_torch_device():
    device = resolve_device("auto")
    summary = describe_device(device)

    assert isinstance(device, torch.device)
    assert summary["device"] == str(device)
    assert summary["cuda_available"] is torch.cuda.is_available()
