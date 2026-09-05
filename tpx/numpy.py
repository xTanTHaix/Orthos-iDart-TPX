from typing import Dict, Any
from tpx import save as _save, load as _load

def save_file(tensors: Dict[str, Any], filename: str, **kwargs: Any) -> None:
    _save(filename, tensors, **kwargs)


def load_file(filename: str, **kwargs: Any) -> Dict[str, Any]:
    return _load(filename, **kwargs)
