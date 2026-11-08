"""App data and inference-metadata tests, without requiring TensorFlow or Streamlit."""
import json
from pathlib import Path
import pytest
from har.app_support import load_split_table, get_dataset_summary, read_json_classes, matrix_from_csv, dataframe_csv_bytes

ROOT = Path(__file__).resolve().parents[1]

def test_bundled_split_statistics():
    rows = load_split_table(ROOT/'data/train_val_test_splits_70_15_15.csv')
    summary = get_dataset_summary(rows)
    assert summary['total_videos'] == 132
    assert summary['split_counts'] == {'train':92,'val':20,'test':20}
    assert summary['classes'] == 7
    assert summary['total_bytes'] == 1862222433
    assert all(min(counts.values())>=2 for counts in summary['by_class'].values())

def test_class_mapping_must_match_trained_checkpoint():
    labels=['clapping','walking','sitting']
    assert read_json_classes(json.dumps(labels)) == labels
    with pytest.raises(ValueError, match='Duplicate'):
        read_json_classes('["clapping","clapping"]')
    with pytest.raises(ValueError, match='do not match'):
        read_json_classes('["unknown"]')

def test_split_roundtrip_csv():
    rows=load_split_table(ROOT/'data/train_val_test_splits_70_15_15.csv')
    assert dataframe_csv_bytes(rows).startswith(b'video_path,activity,bytes,split')

def test_missing_results_are_not_invented(tmp_path):
    assert matrix_from_csv(tmp_path/'nonexistent.csv', ['clapping','walking']) is None
