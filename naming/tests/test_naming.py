"""The vocabulary, and the two things it exists to stop.

One list, shared: a tool that reads less says so beside itself, and nobody
writes the list twice. Both halves are asserted here, because the failure
this package prevents is silent -- a UI offering a format the tool ignores.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sadt_naming as formats


class TestSplitting:
    def test_a_compound_extension_survives_as_one_unit(self):
        """`os.path.splitext` answers ('scan.nii', '.gz'), and no patient is
        called `scan.nii`. This is the whole reason the helper exists."""
        assert formats.split_extension("Pat_0002.nii.gz") == ("Pat_0002", ".nii.gz")
        assert formats.split_extension("Pat_0002.nrrd.gz") == ("Pat_0002", ".nrrd.gz")

    def test_the_longest_match_wins(self):
        """`.nii` comes after `.nii.gz` in the table, and reversing them
        leaves a stem ending in `.nii`."""
        assert formats.split_extension("a.nii") == ("a", ".nii")
        assert formats.split_extension("a.nii.gz")[1] == ".nii.gz"

    def test_another_vocabulary_can_be_asked_for(self):
        assert formats.split_extension("arch.vtk", formats.SURFACE) == ("arch", ".vtk")

    def test_something_the_table_says_nothing_about_still_splits(self):
        """Falling back rather than raising: a name this package has never
        heard of is not an error, it is a name."""
        assert formats.split_extension("notes.docx") == ("notes", ".docx")
        assert formats.split_extension("no_extension") == ("no_extension", "")

    def test_the_case_of_the_name_does_not_decide(self):
        assert formats.split_extension("SCAN.NII.GZ") == ("SCAN", ".NII.GZ")
        assert formats.has_extension("SCAN.NII.GZ", formats.VOLUME)


class TestWriting:
    def test_nrrd_maps_down_rather_than_up(self):
        """ITK has no `.nrrd.gz` writer at all: NRRD compresses inside the
        file. Asking for one is how a run ends in an unwritable path."""
        assert formats.compressed_extension(".nrrd.gz") == ".nrrd"
        assert formats.compressed_extension(".nii") == ".nii.gz"
        assert formats.compressed_extension(".gipl") == ".gipl.gz"

    def test_an_extension_with_no_compressed_spelling_is_returned_as_it_is(self):
        assert formats.compressed_extension(".vtk") == ".vtk"


class TestTheContract:
    def test_this_package_imports_nothing_but_the_standard_library(self):
        """It installs into environments whose pins are deliberately
        incompatible. A dependency here would have to be satisfiable by all
        of them at once, which is the constraint the split removed."""
        source = (Path(formats.__file__)).read_text(encoding="utf-8")
        imports = [line.strip() for line in source.splitlines()
                   if line.startswith(("import ", "from "))]
        assert imports == ["import os"], imports

    def test_a_tool_narrows_by_declaring_less_not_by_editing_this(self):
        """The table says what the FORMAT is. What a tool handles is a fact
        about the tool -- and advertising more than you read is the bug this
        package was created from."""
        reads_only_vtk = (".vtk",)
        assert set(reads_only_vtk) <= set(formats.SURFACE)
        assert formats.has_extension("a.stl", formats.SURFACE)
        assert not formats.has_extension("a.stl", reads_only_vtk)


class TestJaws:
    """The union of every spelling any tool ever accepted.

    Before this, the same clinician's folder passed one tool and was refused
    by the next: `P1_MX.vtk` read by AREG and refused by ASO, `P1_max.vtk`
    read by both and refused by AREG_IOSCBCT, which knew four spellings of
    thirteen.
    """

    def test_every_spelling_any_tool_accepted_is_still_accepted(self):
        for name, jaw in (("P1_U", "Upper"), ("P1_up", "Upper"),
                          ("P1_Upper", "Upper"), ("P1_maxilla", "Upper"),
                          ("P1_max", "Upper"), ("P1_MX", "Upper"),
                          ("P1_L", "Lower"), ("P1_low", "Lower"),
                          ("P1_Lower", "Lower"), ("P1_mandible", "Lower"),
                          ("P1_mandibule", "Lower"), ("P1_mand", "Lower"),
                          ("P1_MD", "Lower")):
            assert formats.jaw_of(name) == jaw, name

    def test_the_french_pair_is_complete(self):
        """`mandibule` was there and `maxillaire` was not, which is an
        oversight rather than a decision about French."""
        assert formats.jaw_of("P1_maxillaire") == "Upper"
        assert formats.jaw_of("P1_mandibule") == "Lower"

    def test_a_name_that_says_no_jaw_says_none_rather_than_defaulting(self):
        """Defaulting to Lower registered a maxillary mesh named
        `patient1.vtk` against the mandibular reference and returned it as a
        success."""
        assert formats.jaw_of("patient1") is None

    def test_a_jaw_word_inside_a_longer_word_is_not_a_jaw(self):
        assert formats.jaw_of("MAXILLOFACIAL_03") is None
        assert formats.jaw_of("P1_lowering") is None

    def test_a_patient_whose_name_is_a_jaw_word_is_a_known_limit(self):
        """Recorded rather than fixed. The rule that would catch it -- a jaw
        token must have something before it -- refuses `Upper_gold.vtk`, the
        published reference's own file name, and both existing tools already
        behave this way."""
        assert formats.jaw_of("MAX_01") == "Upper"
        assert formats.jaw_of("Upper_gold") == "Upper"


class TestMarkers:
    def test_a_marker_only_one_tool_writes_is_stripped_by_all_of_them(self):
        """The union is the point. ASO's own copy lacked `_Seg`, so a cohort
        AMASSS had segmented first read as twice as many patients as it had.
        """
        assert formats.strip_markers("Pat_0002_Seg") == "Pat_0002"
        assert formats.strip_markers("Pat_0002_SegOut") == "Pat_0002"
        assert formats.strip_markers("Pat_0002_OutReg") == "Pat_0002"
        assert formats.strip_markers("Pat_0002_lm_Pred") == "Pat_0002"

    def test_everything_a_marker_introduces_goes_with_it(self):
        """Truncated, not deleted: `P1_Scan_reoriented` is P1, not
        `P1_reoriented`."""
        assert formats.strip_markers("P1_Scan_reoriented") == "P1"
        assert formats.strip_markers("P1_Scan_Seg_Or") == "P1"

    def test_a_marker_inside_a_name_is_not_a_marker(self):
        """`_Seg` is a mark a previous run left; `_Seg1` is part of somebody's
        name. A plain `find` collapsed `P_Seg1_T1` and `P_Seg2_T1` onto one
        patient."""
        assert formats.strip_markers("P_Seg1_T1") == "P_Seg1_T1"
        assert formats.strip_markers("P_Seg2_T1") == "P_Seg2_T1"

    def test_the_case_of_a_marker_decides(self):
        """The spellings the tools actually write. Matching case-insensitively
        would truncate a patient genuinely called `..._SEGMENT`."""
        assert formats.strip_markers("P1_Seg") == "P1"
        assert formats.strip_markers("P1_seg") == "P1"
        assert formats.strip_markers("P1_SEGMENT") == "P1_SEGMENT"

    def test_a_stem_that_is_only_a_marker_keeps_its_name(self):
        """A file called `_Or.nii.gz` is a patient whose name we cannot read,
        not a patient with no name."""
        assert formats.strip_markers("_Or") == "_Or"

    def test_a_timepoint_is_not_a_marker_and_is_never_stripped(self):
        """Two timepoints of one subject are two scans. Collapsing them
        dropped the second and merged both landmark sets into the survivor."""
        assert formats.strip_markers("P1_T1") == "P1_T1"
        assert formats.strip_markers("P1_T2_Or") == "P1_T2"


class TestAJawWordWithATimepointGluedToIt:
    """`A2_UpperT1.vtk` is how upstream's published IOS test set is named.

    The whole-token rule finds no jaw in `UpperT1`, so the mesh was refused and
    the dataset this repository is checked against could not be run at all.
    """

    def test_the_glued_timepoint_is_read_as_the_jaw_it_names(self):
        assert formats.jaw_of("A2_UpperT1") == "Upper"
        assert formats.jaw_of("A2_LowerT2") == "Lower"
        assert formats.jaw_of("A2_UpperT0") == "Upper"

    def test_the_reverse_order_is_a_known_limit(self):
        """`T1Upper` is not read, and that is AREG's limit too. Pinned so it is
        a known one: widening it here alone would put the two vocabularies back
        out of step, which is the whole thing this rule was aligned to fix."""
        assert formats.jaw_of("A2_T1Upper") is None

    def test_a_word_that_merely_starts_with_a_jaw_word_still_names_none(self):
        """The substring disaster, one step removed: a timepoint is evidence,
        the rest of an English word is not."""
        assert formats.jaw_of("P1_lowering") is None
        assert formats.jaw_of("P1_uppermost") is None
        assert formats.jaw_of("MAXILLOFACIAL_03") is None

    def test_every_spelling_the_table_knows_splits_the_same_way(self):
        """The rule is keyed on the two tables, so it covers the short
        spellings too. That is AREG's reading of these names, and one file must
        not be read by AREG and refused here."""
        assert formats.jaw_of("P1_ut1") == "Upper"
        assert formats.jaw_of("P1_lt2") == "Lower"
        assert formats.jaw_of("P1_maxt1") == "Upper"

    def test_an_identifier_that_merely_ends_in_a_timepoint_is_untouched(self):
        """The prefix has to be a jaw as well: `pa` is not one, so `PAT1` stays
        a patient identifier."""
        assert formats.jaw_of("PAT1") is None
        assert formats.jaw_of("SUBJECTT2") is None

    def test_the_whole_token_rule_is_untouched(self):
        assert formats.jaw_of("P1_U_Seg") == "Upper"
        assert formats.jaw_of("Upper_gold") == "Upper"
        assert formats.jaw_of("P1_MX") == "Upper"
