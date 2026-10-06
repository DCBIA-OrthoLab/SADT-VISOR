"""What the class count is read from, and what happens when it cannot be.

Upstream built both heads for 4 classes whatever the checkpoint held, so a
checkpoint trained on another count died inside `load_state_dict` on a shape
mismatch nobody could read. None of this needs the real 176 MB file: the count
lives in one tensor's shape.
"""

import pytest
import torch

from sadt_clic import pipeline


def test_the_count_is_the_length_of_the_classification_bias():
    """One entry per class, background included."""
    state = {"roi_heads.box_predictor.cls_score.bias": torch.zeros(9)}

    assert pipeline.class_count(state) == 9


def test_the_bias_is_preferred_when_both_heads_are_present():
    """They agree in any real checkpoint; if they ever disagree, which one was
    read must not depend on dict ordering."""
    state = {
        "roi_heads.box_predictor.cls_score.weight": torch.zeros(9, 1024),
        "roi_heads.box_predictor.cls_score.bias": torch.zeros(4),
    }

    assert pipeline.class_count(state) == 4


def test_the_weight_is_read_when_the_bias_is_absent():
    """A checkpoint saved without biases still says how many classes it has:
    the weight's first axis is the same count."""
    state = {"roi_heads.box_predictor.cls_score.weight": torch.zeros(6, 1024)}

    assert pipeline.class_count(state) == 6


def test_a_checkpoint_without_the_head_names_what_it_wanted():
    """The message has to say which entry was missing: a caller staging the
    wrong `.pth` can only fix it if it is told what it should have held."""
    with pytest.raises(ValueError) as raised:
        pipeline.class_count({"backbone.body.conv1.weight": torch.zeros(64, 3, 7, 7)})

    assert "roi_heads.box_predictor.cls_score" in str(raised.value)


def test_an_empty_state_dict_is_refused():
    with pytest.raises(ValueError, match="not a Mask R-CNN"):
        pipeline.class_count({})


def test_a_scalar_head_is_refused_rather_than_raising_an_index_error():
    """A 0-dim entry used to reach `shape[0]` and raise `IndexError: tuple
    index out of range`, which names neither the key nor the shape."""
    state = {"roi_heads.box_predictor.cls_score.bias": torch.tensor(4.0)}

    with pytest.raises(ValueError) as raised:
        pipeline.class_count(state)

    assert "roi_heads.box_predictor.cls_score.bias" in str(raised.value)
    assert "()" in str(raised.value)


def test_a_head_with_no_classes_is_refused():
    """A count of 0 would build a predictor with no outputs and fail far away
    from the file that caused it."""
    state = {"roi_heads.box_predictor.cls_score.bias": torch.zeros(0)}

    with pytest.raises(ValueError) as raised:
        pipeline.class_count(state)

    assert "(0,)" in str(raised.value)


def test_the_count_is_a_plain_int_not_a_tensor():
    """It is written into `CLIC_report.json`, and `json.dumps` cannot serialise
    a torch scalar."""
    import json

    classes = pipeline.class_count(
        {"roi_heads.box_predictor.cls_score.bias": torch.zeros(4)}
    )

    assert type(classes) is int
    assert json.dumps({"classes": classes}) == '{"classes": 4}'


def test_a_checkpoint_directory_is_not_globbed_for_a_model(tmp_path):
    """Upstream took `sorted(model_dir.glob("*.pth"))[0]`, so which model
    vintage ran depended on file names. A folder is refused, loudly, rather
    than resolved to the alphabetically first thing in it."""
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "aaa_old.pth").write_bytes(b"")
    (folder / "zzz_new.pth").write_bytes(b"")

    with pytest.raises(IsADirectoryError):
        pipeline.build_model(str(folder), "cpu")


def _save(tmp_path, state):
    path = tmp_path / "model.pth"
    torch.save(state, str(path))
    return str(path)


def test_a_checkpoint_without_the_head_is_a_server_fault_when_loaded(tmp_path):
    """`class_count` refuses with a ValueError, but the checkpoint is installed
    by whoever runs the server: loading it must not be answered as the
    caller's mistake."""
    path = _save(tmp_path, {"backbone.body.conv1.weight": torch.zeros(1)})

    with pytest.raises(RuntimeError) as raised:
        pipeline.build_model(path, "cpu")

    assert not isinstance(raised.value, ValueError)
    assert str(raised.value).startswith("checkpoint is unusable: ")
    assert isinstance(raised.value.__cause__, ValueError)


def test_an_unreadable_checkpoint_is_a_runtime_error(tmp_path):
    path = tmp_path / "model.pth"
    path.write_bytes(b"these bytes are not a torch checkpoint")

    with pytest.raises(RuntimeError) as raised:
        pipeline.build_model(str(path), "cpu")

    assert str(raised.value).startswith("checkpoint could not be read (")


def test_weights_of_the_wrong_shape_put_the_cause_first(tmp_path):
    """torch lists every mismatched tensor on its own line; the operator sees
    the first few hundred characters, so the cause has to lead."""
    path = _save(tmp_path, {"roi_heads.box_predictor.cls_score.bias": torch.zeros(4)})

    with pytest.raises(RuntimeError) as raised:
        pipeline.build_model(path, "cpu")

    message = str(raised.value)
    assert message.startswith("checkpoint weights do not fit the Mask R-CNN layout (")
    assert "\n" not in message


def test_a_missing_checkpoint_stays_a_file_not_found_error(tmp_path):
    """A wrong name is the caller's to fix, and stays answered as such."""
    with pytest.raises(FileNotFoundError):
        pipeline.build_model(str(tmp_path / "absent.pth"), "cpu")
