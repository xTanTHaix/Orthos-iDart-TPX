from typing import Dict, Any, Optional
from tpx import open as _open


def save_file(tensors: Dict[str, Any], filename: str, **kwargs: Any) -> None:
    numpy_tensors = {}
    for k, v in tensors.items():
        if hasattr(v, 'detach'):
            v = v.detach()
        if hasattr(v, 'cpu'):
            v = v.cpu()
        if hasattr(v, 'numpy'):
            numpy_tensors[str(k)] = v.numpy()
        else:
            numpy_tensors[str(k)] = v

    with _open(filename, mode='w+', **kwargs) as db:
        db.put_batch(numpy_tensors)
        db.flush()


def load_file(filename: str, device: str = 'cpu', **kwargs: Any) -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        raise ImportError('PyTorch is required for tpx.torch. Install torch with pip install torch')

    with _open(filename, mode='r', **kwargs) as db:
        all_keys = db.keys()
        batch = db.get_batch(all_keys)
        results = {}
        for k, arr in batch.items():
            t = torch.from_numpy(arr)
            if device != 'cpu':
                t = t.to(device)
            results[k] = t
        return results
