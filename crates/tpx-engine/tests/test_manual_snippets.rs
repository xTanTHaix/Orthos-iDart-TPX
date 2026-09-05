use tpx_core::types::DTypeId;
use tpx_engine::database::TPXDatabase;

#[test]
fn test_section_0_2_rust_snippet() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let path = temp_dir.path().join("storage_vault.tpx");

    // 1. สร้างไฟล์ฐานข้อมูล TPX
    let db = TPXDatabase::create(&path, None)?;

    // 2. เขียนข้อมูลดิบระดับไบต์พร้อมระบุ Data Type และ Shape
    let raw_floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: &[u8] = bytemuck::cast_slice(&raw_floats);
    db.put(
        "features/layer_1",
        bytes,
        DTypeId::Float32,
        &[4],
        None,
        None,
    )?;

    // หรือเขียนเป็นชุดแบบขนานระดับมัลติคอร์ (Vectorized Parallel Bulk Write)
    let batch = vec![
        ("features/layer_2", bytes, DTypeId::Float32, &[4][..]),
        ("features/layer_3", bytes, DTypeId::Float32, &[4][..]),
    ];
    db.put_batch(&batch)?;

    // 3. อ่านข้อมูลกลับคืนมา
    if let Some(tensor) = db.get("features/layer_1")? {
        assert_eq!(tensor.dtype, DTypeId::Float32);
        assert_eq!(tensor.shape, vec![4]);
        assert_eq!(tensor.data.len(), 16);
    } else {
        panic!("Failed to read features/layer_1");
    }

    // 4. บันทึก Checkpoint และปิดไฟล์อย่างปลอดภัย
    db.flush()?;
    db.close()?;
    Ok(())
}

#[test]
fn test_rust_one_liners() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let path = temp_dir.path().join("oneliner.tpx");

    // Test TPXDatabase::save_tensor
    let raw: Vec<f32> = vec![10.0, 20.0, 30.0];
    let bytes: &[u8] = bytemuck::cast_slice(&raw);
    TPXDatabase::save_tensor(&path, "weights/w0", bytes, DTypeId::Float32, &[3])?;

    // Test TPXDatabase::list_keys
    let keys = TPXDatabase::list_keys(&path)?;
    assert_eq!(keys, vec!["weights/w0".to_string()]);

    // Test TPXDatabase::load_tensor
    let loaded = TPXDatabase::load_tensor(&path, "weights/w0")?;
    assert!(loaded.is_some());
    let t = loaded.unwrap();
    assert_eq!(t.shape, vec![3]);
    let floats: &[f32] = bytemuck::cast_slice(&t.data);
    assert_eq!(floats, &[10.0, 20.0, 30.0]);

    // Test TPXDatabase::save_batch
    let path_batch = temp_dir.path().join("batch.tpx");
    let batch = vec![
        ("layer/1", bytes, DTypeId::Float32, &[3][..]),
        ("layer/2", bytes, DTypeId::Float32, &[3][..]),
    ];
    TPXDatabase::save_batch(&path_batch, &batch)?;
    let batch_keys = TPXDatabase::list_keys(&path_batch)?;
    assert_eq!(batch_keys.len(), 2);

    Ok(())
}
