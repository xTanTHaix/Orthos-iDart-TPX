use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::HashMap;
use std::path::PathBuf;
use tpx_core::attrs::AttrValue;
use tpx_core::types::{DTypeId, OpenMode};
use tpx_engine::database::TPXDatabase;

#[pyclass]
pub struct NativeTPXEngine {
    db: TPXDatabase,
}

#[pymethods]
impl NativeTPXEngine {
    #[new]
    #[pyo3(signature = (path, mode="a", enable_direct_io=true, alignment=4096, enable_sqpoll=false, enable_iopoll=false, enable_spdk=false, enable_hugepages=false, enable_hw_offload=false, shard_count=None))]
    pub fn new(
        path: String,
        mode: &str,
        enable_direct_io: bool,
        alignment: u32,
        enable_sqpoll: bool,
        enable_iopoll: bool,
        enable_spdk: bool,
        enable_hugepages: bool,
        enable_hw_offload: bool,
        shard_count: Option<u16>,
    ) -> PyResult<Self> {
        let _ = (
            enable_direct_io,
            alignment,
            enable_sqpoll,
            enable_iopoll,
            enable_spdk,
            enable_hugepages,
            enable_hw_offload,
        );

        let open_mode = match mode {
            "r" => OpenMode::ReadOnly,
            "w" | "w+" => OpenMode::ReadWrite,
            _ => OpenMode::Append,
        };

        let path_buf = PathBuf::from(&path);
        let db = if (open_mode == OpenMode::ReadWrite || open_mode == OpenMode::Append)
            && !path_buf.exists()
        {
            TPXDatabase::create(&path_buf, shard_count)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else {
            TPXDatabase::open(&path_buf, open_mode)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        };

        Ok(Self { db })
    }

    #[pyo3(signature = (key, data, dtype, shape, attrs=None, retain=None))]
    pub fn put(
        &self,
        key: &str,
        data: &Bound<'_, PyBytes>,
        dtype: u8,
        shape: Vec<u32>,
        attrs: Option<HashMap<String, String>>,
        retain: Option<String>,
    ) -> PyResult<()> {
        let dtype_id = DTypeId::from_u8(dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let mut attrs_map = HashMap::new();
        if let Some(m) = attrs {
            for (k, v) in m {
                attrs_map.insert(k, AttrValue::String(v));
            }
        }

        self.db
            .put(
                key,
                data.as_bytes(),
                dtype_id,
                &shape,
                Some(attrs_map),
                retain.as_deref(),
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[pyo3(signature = (tensors))]
    pub fn put_batch(
        &self,
        tensors: Vec<(String, Bound<'_, PyBytes>, u8, Vec<u32>)>,
    ) -> PyResult<()> {
        let mut prepared = Vec::with_capacity(tensors.len());
        for (key, data, dtype, shape) in &tensors {
            let dtype_id = DTypeId::from_u8(*dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            prepared.push((key.as_str(), data.as_bytes(), dtype_id, shape.as_slice()));
        }

        self.db
            .put_batch(&prepared)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    pub fn get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.db.get(key) {
            Ok(Some(tensor)) => {
                let dict = PyDict::new(py);
                dict.set_item("dtype", tensor.dtype as u8)?;
                dict.set_item("shape", tensor.shape)?;
                dict.set_item("data", PyBytes::new(py, &tensor.data))?;
                Ok(Some(dict))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    pub fn get_prefix<'py>(&self, py: Python<'py>, prefix: &str) -> PyResult<Bound<'py, PyDict>> {
        let tensors = self
            .db
            .get_prefix(prefix)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let out = PyDict::new(py);
        for (k, tensor) in tensors {
            let item = PyDict::new(py);
            item.set_item("dtype", tensor.dtype as u8)?;
            item.set_item("shape", tensor.shape)?;
            item.set_item("data", PyBytes::new(py, &tensor.data))?;
            out.set_item(k, item)?;
        }
        Ok(out)
    }

    pub fn get_batch<'py>(
        &self,
        py: Python<'py>,
        keys: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let results = self
            .db
            .get_batch(&key_refs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let out = PyDict::new(py);
        for (k, tensor) in results {
            let item = PyDict::new(py);
            item.set_item("dtype", tensor.dtype as u8)?;
            item.set_item("shape", tensor.shape)?;
            item.set_item("data", PyBytes::new(py, &tensor.data))?;
            out.set_item(k, item)?;
        }
        Ok(out)
    }

    pub fn keys(&self) -> PyResult<Vec<String>> {
        self.db
            .keys()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    pub fn delete(&self, key: &str) -> PyResult<()> {
        self.db
            .delete(key)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    pub fn list_versions(&self, key: &str) -> PyResult<Vec<u32>> {
        self.db
            .list_versions(key)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    pub fn get_version<'py>(
        &self,
        py: Python<'py>,
        key: &str,
        version: u32,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.db.get_version(key, version) {
            Ok(Some(tensor)) => {
                let dict = PyDict::new(py);
                dict.set_item("dtype", tensor.dtype as u8)?;
                dict.set_item("shape", tensor.shape)?;
                dict.set_item("data", PyBytes::new(py, &tensor.data))?;
                Ok(Some(dict))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    pub fn flush(&self) -> PyResult<()> {
        self.db
            .flush()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    pub fn compact(&self) -> PyResult<()> {
        self.db
            .compact()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    pub fn close(&self) -> PyResult<()> {
        self.db
            .close()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

#[pymodule]
fn _native(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeTPXEngine>()?;
    Ok(())
}
