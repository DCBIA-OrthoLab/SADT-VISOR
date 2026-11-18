"""The two shapes a transform arrives in, and what happens to everything else.

`sitk.ReadTransform` reads ITK's own formats. Greedy writes a bare 4x4 matrix
in text and calls it `.mat`, so does upstream's own `writeIdentityInit`, and
`ReadTransform` refuses it with a MatlabTransformIO error -- which is why
AutoMatrix could not consume what GreedyReg produced. The plain matrix is
therefore a fallback, not a branch on the extension.
"""

import json

import pytest
import SimpleITK as sitk

from sadt_automatrix import pipeline


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


# ---------------------------------------------------------------------------
# What must be read
# ---------------------------------------------------------------------------

def test_an_itk_transform_written_into_a_mat_is_read_by_itk(tmp_path):
    """ITK's own MATLAB writer produces a binary `.mat` that `ReadTransform`
    reads. The extension says nothing about which of the two shapes it is."""
    path = tmp_path / "P1_transform.mat"
    sitk.WriteTransform(sitk.TranslationTransform(3, (5.0, 0.0, 0.0)), str(path))

    transform = pipeline.read_transform(str(path))

    assert transform.TransformPoint((0.0, 0.0, 0.0)) == (5.0, 0.0, 0.0)


def test_an_itk_affine_survives_the_round_trip(tmp_path):
    path = tmp_path / "P1_transform.tfm"
    affine = sitk.AffineTransform(3)
    affine.SetMatrix([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    affine.SetTranslation([1.0, 2.0, 3.0])
    sitk.WriteTransform(affine, str(path))

    assert pipeline.read_transform(str(path)).TransformPoint((2.0, 0.0, 0.0)) == (1.0, 4.0, 3.0)


def test_a_plain_matrix_carries_its_rotation_and_its_translation(tmp_path):
    """Both halves, not just the offset: the first three columns are the
    rotation, the fourth is the translation."""
    path = tmp_path / "P1_transform.mat"
    path.write_text("0 -1 0 1\n1 0 0 2\n0 0 1 3\n0 0 0 1\n")

    transform = pipeline.read_transform(str(path))

    assert transform.GetMatrix() == (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert transform.GetTranslation() == (1.0, 2.0, 3.0)
    assert transform.TransformPoint((2.0, 0.0, 0.0)) == (1.0, 4.0, 3.0)


def test_a_plain_matrix_without_a_trailing_newline_is_read(tmp_path):
    """A writer that does not end its last line must not cost a patient."""
    path = tmp_path / "P1_transform.mat"
    path.write_text("1 0 0 3.5\n0 1 0 0\n0 0 1 0\n0 0 0 1")

    assert pipeline.read_transform(str(path)).TransformPoint((0.0, 0.0, 0.0)) == (3.5, 0.0, 0.0)


def test_blank_lines_between_the_rows_are_ignored(tmp_path):
    path = tmp_path / "P1_transform.mat"
    path.write_text("\n1 0 0 3.5\n\n0 1 0 0\n\n0 0 1 0\n\n0 0 0 1\n\n")

    assert pipeline.read_transform(str(path)).TransformPoint((0.0, 0.0, 0.0)) == (3.5, 0.0, 0.0)


@pytest.mark.parametrize("extension", [".mat", ".tfm", ".h5", ".txt"])
def test_the_extension_does_not_decide_which_reader_runs(tmp_path, extension):
    """The fallback is tried for every extension, because greedy's file name
    says nothing about its contents and neither does anyone else's."""
    path = tmp_path / f"P1_transform{extension}"
    path.write_text("1 0 0 3.5\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    assert pipeline.read_transform(str(path)).TransformPoint((0.0, 0.0, 0.0)) == (3.5, 0.0, 0.0)


@pytest.mark.parametrize("text", [
    "1.0 0.0 0.0 3.5\n0.0 1.0 0.0 0.0\n0.0 0.0 1.0 0.0\n0.0 0.0 0.0 1.0\n",
    "1 0 0 3.5\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
    "1e0 0 0 3.5e0\n0 1e0 0 0\n0 0 1e0 0\n0 0 0 1e0\n",
    "  1\t0  0   3.5 \n 0 1 0 0\n0 0 1 0\n0 0 0 1\n",
])
def test_the_numbers_may_be_written_any_way_python_reads_a_float(tmp_path, text):
    path = tmp_path / "P1_transform.mat"
    path.write_text(text)

    assert pipeline.read_transform(str(path)).TransformPoint((0.0, 0.0, 0.0)) == (3.5, 0.0, 0.0)


# ---------------------------------------------------------------------------
# What must be refused, and how
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,text", [
    # 16 numbers on one line: refused rather than reshaped, because a flat list
    # is row-major or column-major depending on who wrote it and guessing
    # transposes the rotation in silence.
    ("one line", "1 0 0 3.5 0 1 0 0 0 0 1 0 0 0 0 1\n"),
    ("empty", ""),
    ("blank lines only", "\n\n\n"),
    ("3x3", "1 0 0\n0 1 0\n0 0 1\n"),
    ("5x5", "1 0 0 0 0\n0 1 0 0 0\n0 0 1 0 0\n0 0 0 1 0\n0 0 0 0 1\n"),
    ("3 rows of 4", "1 0 0 1\n0 1 0 2\n0 0 1 3\n"),
    ("ragged", "1 0 0 1\n0 1 0\n0 0 1 3\n0 0 0 1\n"),
    ("prose", "this is not a transform\n"),
    ("json", '{"transform": "identity"}\n'),
])
def test_a_file_that_is_neither_shape_is_refused(tmp_path, label, text):
    path = tmp_path / "P1_transform.mat"
    path.write_text(text)

    with pytest.raises(RuntimeError):
        pipeline.read_transform(str(path))


def test_the_refusal_names_the_file_and_the_shape_it_wanted(tmp_path):
    """A run collects this message per patient, so it has to be readable on its
    own: which file, and what would have been accepted."""
    path = tmp_path / "C_0001_CB_Reg_transform.mat"
    path.write_text("nothing like a transform\n")

    with pytest.raises(RuntimeError) as failure:
        pipeline.read_transform(str(path))

    message = str(failure.value)
    assert "C_0001_CB_Reg_transform.mat" in message
    assert "4x4 matrix" in message
    assert "ITK transform" in message
    # And what ITK itself said, so the real cause is not thrown away.
    assert "ITK said:" in message


def test_binary_junk_is_refused_rather_than_raising_a_decode_error(tmp_path):
    """The fallback opens the file as text; an ITK-looking binary that ITK
    cannot read must still come back as the same clear refusal."""
    path = tmp_path / "P1_transform.mat"
    path.write_bytes(bytes(range(256)))

    with pytest.raises(RuntimeError, match="neither an ITK transform nor a 4x4 matrix"):
        pipeline.read_transform(str(path))


def test_a_transform_that_is_not_there_says_so(tmp_path):
    """FileNotFoundError, naming the path -- not the "neither shape" message,
    which would blame the contents of a file nobody wrote."""
    with pytest.raises(FileNotFoundError, match="P1_transform.tfm"):
        pipeline.read_transform(str(tmp_path / "P1_transform.tfm"))


# ---------------------------------------------------------------------------
# A matrix that reads but cannot be inverted
# ---------------------------------------------------------------------------

def test_a_singular_matrix_is_read_because_reading_is_not_inverting(tmp_path):
    """Resampling an image needs the forward transform only, so this must not
    be refused at read time."""
    path = tmp_path / "P1_transform.mat"
    path.write_text("0 0 0 1\n0 0 0 2\n0 0 0 3\n0 0 0 1\n")

    transform = pipeline.read_transform(str(path))

    assert transform.TransformPoint((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)


def test_a_singular_matrix_fails_where_the_inverse_is_actually_needed(tmp_path):
    """Which is the landmark path, and it fails loudly. Upstream logged and
    returned, so the output was the input file copied and called a result."""
    path = tmp_path / "P1_transform.mat"
    path.write_text("0 0 0 1\n0 0 0 2\n0 0 0 3\n0 0 0 1\n")
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": "A", "position": [1.0, 2.0, 3.0], "positionStatus": "defined"}]}]}))

    # A ValueError: the matrix is what the caller sent, theirs to fix.
    with pytest.raises(ValueError, match="no inverse"):
        pipeline.apply_to_landmarks(
            str(source), pipeline.read_transform(str(path)),
            str(tmp_path / "out.mrk.json"),
        )

    assert not (tmp_path / "out.mrk.json").exists(), "a file was written anyway"


# ---------------------------------------------------------------------------
# What counts as a transform at all
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("P1_transform.tfm", True),
    ("P1_transform.mat", True),
    ("P1_transform.h5", True),
    ("P1_transform.hdf5", True),
    ("P1_transform.txt", True),
    ("P1_TRANSFORM.TFM", True),
    ("P1_T1.nii.gz", False),
    ("P1_lm.mrk.json", False),
    ("notes.md", False),
])
def test_what_counts_as_a_transform_file(name, expected):
    assert pipeline.is_transform_file(name) is expected


def test_the_identity_matrix_moves_nothing(tmp_path, matrix_file):
    path = matrix_file(tmp_path / "P1_transform.mat", IDENTITY)

    assert pipeline.read_transform(str(path)).TransformPoint((7.0, 8.0, 9.0)) == (7.0, 8.0, 9.0)
