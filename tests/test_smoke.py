"""Fast CPU-only smoke tests. Run from repo root:  python -m pytest tests/ -q.

Tests touching train.py / stratified splits are skipped when sklearn is not
installed (train.py imports sklearn at module level).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip('torch')
PIL_Image = pytest.importorskip('PIL.Image')


def _needs_sklearn():
    return pytest.importorskip('sklearn')


def test_seed_everything_deterministic():
    from dataset import seed_everything
    import random
    seed_everything(123)
    a = [random.random() for _ in range(5)]
    seed_everything(123)
    b = [random.random() for _ in range(5)]
    assert a == b
    seed_everything(999)
    assert [random.random() for _ in range(5)] != a


def test_stratified_split_disjoint_and_covering():
    _needs_sklearn()
    from dataset import stratified_train_val_test_split
    labels = [0] * 20 + [1] * 20 + [2] * 20 + [3] * 20
    tr, va, te = stratified_train_val_test_split(labels, seed=7)
    assert len(tr) + len(va) + len(te) == len(labels)
    assert not (set(tr) & set(va) & set(te))
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    # Same seed -> identical split (reproducibility)
    tr2, va2, te2 = stratified_train_val_test_split(labels, seed=7)
    assert tr == tr2 and va == va2 and te == te2


def test_compute_class_weights_balanced():
    _needs_sklearn()
    from train import compute_class_weights
    w = compute_class_weights([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
    assert w.shape == (4,)
    assert torch.allclose(w, torch.ones(4))


def test_compute_class_weights_imbalanced_upweights_minority():
    _needs_sklearn()
    from train import compute_class_weights
    w = compute_class_weights([0] * 100 + [1] * 10 + [2] * 10 + [3] * 10)
    assert w[0].item() < w[1].item()


def test_amp_dtype_mapping():
    _needs_sklearn()
    from train import _amp_dtype_for, _make_scaler
    assert _amp_dtype_for(torch.device('xpu')) == torch.bfloat16
    assert _amp_dtype_for(torch.device('cpu')) == torch.bfloat16
    assert _amp_dtype_for(torch.device('cuda')) == torch.float16
    assert _make_scaler(torch.device('xpu'), True) is None
    assert _make_scaler(torch.device('cpu'), True) is None


def test_classifier_forward_shape():
    from models import DenseNet121TumorClassificationModel
    m = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 1, 64, 64))
    assert tuple(out.shape) == (2, 4)


def test_autoencoder_forward_shapes():
    from models import EnhancedAutoencoder
    m = EnhancedAutoencoder()
    m.eval()
    with torch.no_grad():
        recon, att = m(torch.randn(2, 1, 64, 64))
    assert tuple(recon.shape) == (2, 1, 64, 64)
    assert att.shape[0] == 2 and att.shape[1] == 1


def test_default_loader_kwargs_xpu_safe():
    from dataset import default_loader_kwargs
    kw = default_loader_kwargs(batch_size=32, device=torch.device('xpu'))
    assert kw['num_workers'] == 0
    assert 'pin_memory' not in kw  # unified memory: pinning is pure overhead
    kw2 = default_loader_kwargs(batch_size=16, num_workers=4, device=torch.device('xpu'))
    assert kw2['num_workers'] == 4
    assert kw2['persistent_workers'] is True


def test_mri_dataset_indices_subset(tmp_path):
    import numpy as np
    from dataset import MRIDataset
    for cls in ['glioma', 'meningioma', 'notumor', 'pituitary']:
        d = tmp_path / cls
        d.mkdir()
        for i in range(4):
            PIL_Image.fromarray((np.random.rand(32, 32) * 255).astype('uint8')).save(d / f'{i}.png')
    full = MRIDataset(str(tmp_path), image_size=(32, 32), split='full', apply_normalize=False)
    assert len(full) == 16
    sub = MRIDataset(str(tmp_path), image_size=(32, 32), split='full', apply_normalize=False, indices=[0, 1, 2])
    assert len(sub) == 3
    img, label = sub[0]
    assert tuple(img.shape) == (1, 32, 32)
    assert 0 <= int(label) <= 3


def test_reporting_generates_html(tmp_path):
    from reporting import generate_diagnostic_report
    orig = PIL_Image.new('L', (64, 64), 128)
    out = str(tmp_path / 'report.html')
    generate_diagnostic_report('scan.png', 'glioma', 0.93,
                               {'total_area_cm2': 1.5, 'width_cm': 1.0, 'height_cm': 1.2, 'has_tumor': True},
                               orig, orig, None, out, 'cpu',
                               patient_name='Test', patient_id='T-1', age=30, gender='Other',
                               physician_notes='note')
    assert os.path.exists(out)
    html = open(out, encoding='utf-8').read()
    assert 'GLIOMA' in html and 'Test' in html


def test_database_roundtrip(tmp_path):
    pytest.importorskip('pandas')
    from database import log_diagnostic_run, fetch_diagnostic_records, delete_diagnostic_record
    db = str(tmp_path / 't.db')
    log_diagnostic_run('f.png', 'glioma', 0.9,
                       {'total_area_cm2': 1.0, 'width_cm': 1.0, 'height_cm': 1.0},
                       None, patient_name='T', db_path=db)
    df = fetch_diagnostic_records(db_path=db)
    assert len(df) == 1 and df.iloc[0]['pred_class'] == 'glioma'
    assert delete_diagnostic_record(int(df.iloc[0]['id']), db_path=db) is True
    assert len(fetch_diagnostic_records(db_path=db)) == 0
