import pytest
import numpy as np
import os
import tpx

def test_tpx_one_liners_and_dict_interface(tmp_path):
    file_path = str(tmp_path / 'test_ergonomics.tpx')
    
    # 1. Test tpx.save with a dictionary of arrays
    tensors = {
        'weight': np.arange(12, dtype=np.float32).reshape(3, 4),
        'bias': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'scalar_int': np.int64(42),
    }
    tpx.save(file_path, tensors)
    assert os.path.exists(file_path)

    # 2. Test tpx.keys
    keys = tpx.keys(file_path)
    assert set(keys) == {'weight', 'bias', 'scalar_int'}

    # 3. Test tpx.info
    info = tpx.info(file_path)
    assert info['tensor_count'] == 3
    assert info['file_size_bytes'] > 0
    assert set(info['keys']) == {'weight', 'bias', 'scalar_int'}

    # 4. Test tpx.load (all tensors)
    loaded = tpx.load(file_path)
    assert len(loaded) == 3
    np.testing.assert_array_equal(loaded['weight'], tensors['weight'])
    np.testing.assert_array_equal(loaded['bias'], tensors['bias'])
    assert loaded['scalar_int'] == np.int64(42)

    # 5. Test tpx.load (selective single key)
    loaded_weight = tpx.load(file_path, 'weight')
    np.testing.assert_array_equal(loaded_weight, tensors['weight'])

    # 6. Test dict-like interface on TPXDatabase
    with tpx.open(file_path, mode='a') as db:
        assert 'weight' in db
        assert 'nonexistent' not in db
        assert len(db) == 3
        
        # Test __getitem__
        w = db['weight']
        np.testing.assert_array_equal(w, tensors['weight'])
        
        # Test __setitem__
        db['new_layer'] = np.ones((2, 2), dtype=np.float32)
        assert 'new_layer' in db
        assert len(db) == 4
        
        # Test keys(), items(), values()
        all_k = db.keys()
        assert 'new_layer' in all_k
        items = dict(db.items())
        assert 'new_layer' in items
        np.testing.assert_array_equal(items['new_layer'], np.ones((2, 2), dtype=np.float32))

def test_safetensors_numpy_dropin(tmp_path):
    from tpx.numpy import save_file, load_file
    
    file_path = str(tmp_path / 'test_numpy_dropin.tpx')
    data = {
        'x': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'y': np.array([[10, 20], [30, 40]], dtype=np.int32),
    }
    save_file(data, file_path)
    loaded = load_file(file_path)
    
    assert set(loaded.keys()) == {'x', 'y'}
    np.testing.assert_array_equal(loaded['x'], data['x'])
    np.testing.assert_array_equal(loaded['y'], data['y'])

def test_safetensors_torch_dropin(tmp_path):
    torch = pytest.importorskip('torch')
    from tpx.torch import save_file, load_file
    
    file_path = str(tmp_path / 'test_torch_dropin.tpx')
    tensors = {
        'conv.weight': torch.randn(8, 4, 3, 3),
        'fc.bias': torch.zeros(16),
    }
    save_file(tensors, file_path)
    
    # Load onto CPU
    loaded = load_file(file_path, device='cpu')
    assert set(loaded.keys()) == {'conv.weight', 'fc.bias'}
    assert torch.allclose(loaded['conv.weight'], tensors['conv.weight'])
    assert torch.allclose(loaded['fc.bias'], tensors['fc.bias'])
