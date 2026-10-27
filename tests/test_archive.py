import gc
import os
import zipfile

import numpy as np
import pytest

import wfdb
from wfdb.io import archive as wfdb_archive_module
from wfdb.io.archive import WFDBArchive, create_archive


def record_files(record_name):
    """
    Return the header and signal file paths of a record on disk.
    """
    header = wfdb.rdheader(record_name)
    dir_name = os.path.dirname(record_name)
    files = [record_name + ".hea"]
    for file_name in dict.fromkeys(header.file_name or []):
        files.append(os.path.join(dir_name, file_name))
    return files


def archive_sample_record(record_name, tmp_path):
    """
    Put a record from sample-data into a .wfdb archive and return the
    archive path.
    """
    archive_path = os.fspath(tmp_path / (os.path.basename(record_name) + ".wfdb"))
    create_archive(record_files(record_name), archive_path)
    return archive_path


def assert_records_equal(record, expected):
    assert record.n_sig == expected.n_sig
    assert record.sig_len == expected.sig_len
    assert record.sig_name == expected.sig_name
    assert record.fmt == expected.fmt
    for attr in ["d_signal", "p_signal"]:
        if getattr(expected, attr) is not None:
            np.testing.assert_array_equal(
                getattr(record, attr), getattr(expected, attr)
            )
    for attr in ["e_d_signal", "e_p_signal"]:
        if getattr(expected, attr) is not None:
            for got, want in zip(getattr(record, attr), getattr(expected, attr)):
                np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "record_name",
    [
        "sample-data/100",
        "sample-data/test01_00s",
        "sample-data/flacformats",
        "sample-data/a103l",
    ],
)
@pytest.mark.parametrize("physical", [True, False])
def test_read_archive_matches_files(record_name, physical, tmp_path):
    """
    Reading a record from an archive gives the same result as reading the
    record files directly.
    """
    archive_path = archive_sample_record(record_name, tmp_path)

    record = wfdb.rdrecord(archive_path, physical=physical)
    expected = wfdb.rdrecord(record_name, physical=physical)

    assert_records_equal(record, expected)


def test_read_archive_expanded(tmp_path):
    """
    Signals with several samples per frame can be read without smoothing.
    """
    record_name = "sample-data/test01_00s_frame"
    archive_path = archive_sample_record(record_name, tmp_path)

    record = wfdb.rdrecord(archive_path, smooth_frames=False)
    expected = wfdb.rdrecord(record_name, smooth_frames=False)

    assert_records_equal(record, expected)


def test_read_archive_channels_and_range(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    record = wfdb.rdrecord(archive_path, sampfrom=1000, sampto=3000, channels=[1])
    expected = wfdb.rdrecord(
        "sample-data/100", sampfrom=1000, sampto=3000, channels=[1]
    )

    assert_records_equal(record, expected)


def test_rdsamp_and_rdheader_archive(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    signal, fields = wfdb.rdsamp(archive_path, sampto=500)
    expected_signal, expected_fields = wfdb.rdsamp("sample-data/100", sampto=500)
    np.testing.assert_array_equal(signal, expected_signal)
    assert fields == expected_fields

    header = wfdb.rdheader(archive_path)
    assert header.record_name == "100"
    assert header.sig_name == expected_fields["sig_name"]


def test_read_archive_without_sig_len(tmp_path):
    """
    The signal length is inferred from the size of the dat file in the
    archive when the header doesn't give it.
    """
    expected = wfdb.rdrecord("sample-data/100")
    with open("sample-data/100.hea") as f:
        lines = f.read().splitlines()
    # Drop sig_len (and anything after it) from the record line
    lines[0] = " ".join(lines[0].split()[:3])
    archive_path = os.fspath(tmp_path / "100.wfdb")
    with WFDBArchive(archive_path, mode="w") as archive:
        archive.write("100.hea", ("\n".join(lines) + "\n").encode())
        archive.add_file("sample-data/100.dat")

    record = wfdb.rdrecord(archive_path)

    assert record.sig_len == expected.sig_len
    np.testing.assert_array_equal(record.p_signal, expected.p_signal)


def test_read_archive_twice(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    first = wfdb.rdrecord(archive_path, sampto=100)
    second = wfdb.rdrecord(archive_path, sampto=100)

    np.testing.assert_array_equal(first.p_signal, second.p_signal)


def test_read_replaced_archive(tmp_path):
    """
    Replacing an archive and reading it again returns the new record.
    """
    archive_path = archive_sample_record("sample-data/100", tmp_path)
    assert wfdb.rdrecord(archive_path).sig_name == ["MLII", "V5"]

    create_archive(record_files("sample-data/test01_00s"), archive_path)

    record = wfdb.rdrecord(archive_path)
    expected = wfdb.rdrecord("sample-data/test01_00s")
    assert_records_equal(record, expected)


def test_read_archive_header_named_differently(tmp_path):
    """
    An archive with a single header file can be read even if the header
    isn't named after the archive.
    """
    archive_path = os.fspath(tmp_path / "renamed.wfdb")
    create_archive(record_files("sample-data/100"), archive_path)

    record = wfdb.rdrecord(archive_path, sampto=100)

    assert record.record_name == "100"


def test_read_archive_several_headers(tmp_path):
    """
    If an archive made by other software isn't named after one of its
    headers, and it contains several, it is not clear which record to
    read.
    """
    archive_path = os.fspath(tmp_path / "both.wfdb")
    with zipfile.ZipFile(archive_path, "w") as zf:
        for path in record_files("sample-data/100") + ["sample-data/test01_00s.hea"]:
            zf.write(path, arcname=os.path.basename(path))

    with pytest.raises(FileNotFoundError, match="several header files"):
        wfdb.rdrecord(archive_path)


def test_read_archive_missing_dat_file(tmp_path):
    """
    A dat file named in the header that is not in the archive is an error,
    even if the archive contains another dat file.
    """
    with open("sample-data/100.hea") as f:
        header = f.read().replace("100.dat", "other.dat")
    archive_path = os.fspath(tmp_path / "100.wfdb")
    with WFDBArchive(archive_path, mode="w") as archive:
        archive.write("100.hea", header.encode())
        archive.add_file("sample-data/100.dat")

    with pytest.raises(FileNotFoundError, match="other.dat"):
        wfdb.rdrecord(archive_path)


def test_read_archive_without_header(tmp_path):
    archive_path = os.fspath(tmp_path / "100.wfdb")
    create_archive(["sample-data/100.dat"], archive_path)

    with pytest.raises(FileNotFoundError, match="100.hea"):
        wfdb.rdrecord(archive_path)


def test_read_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="Archive not found"):
        wfdb.rdrecord(os.fspath(tmp_path / "missing.wfdb"))


def test_read_invalid_archive(tmp_path):
    archive_path = tmp_path / "invalid.wfdb"
    archive_path.write_bytes(b"not a zip file")

    with pytest.raises(ValueError, match="Not a valid WFDB archive"):
        wfdb.rdrecord(os.fspath(archive_path))


def test_read_archive_header_too_large(tmp_path, monkeypatch):
    archive_path = archive_sample_record("sample-data/100", tmp_path)
    monkeypatch.setattr(wfdb_archive_module, "MAX_HEADER_SIZE", 10)

    with pytest.raises(ValueError, match="larger than 10 bytes"):
        wfdb.rdrecord(archive_path)


def multi_record_files(record_name):
    """
    Return the paths of the files of a multi-segment record on disk: its
    header, and the header and dat files of each segment. Segments that
    are not in sample-data are left out.
    """
    dir_name = os.path.dirname(record_name)
    files = [record_name + ".hea"]
    for seg_name in wfdb.rdheader(record_name).seg_name:
        seg_path = os.path.join(dir_name, seg_name)
        if seg_name != "~" and os.path.exists(seg_path + ".hea"):
            files += [path for path in record_files(seg_path) if not path.endswith("~")]
    return list(dict.fromkeys(files))


MULTI_SEGMENT_READS = [
    # Fixed layout, segments sharing one dat file
    ("fixed1/v102s", {}),
    ("fixed1/v102s", dict(sampfrom=70000, sampto=80000, channels=[1, 0, 3])),
    # Fixed layout, one dat file per segment
    ("041s/041s", dict(channels=[3, 2, 1])),
    # Variable layout with a layout header and empty segments
    ("s25047/s25047-2704-05-04-10-44", {}),
    ("s25047/s25047-2704-05-04-10-44", dict(channel_names=["ABP", "II"])),
    (
        "s00001/s00001-2896-10-10-00-31",
        dict(sampfrom=14428364, sampto=14428375),
    ),
]


@pytest.mark.parametrize("record, kwargs", MULTI_SEGMENT_READS)
def test_read_multi_segment_archive_matches_files(record, kwargs, tmp_path):
    """
    Reading a multi-segment record from an archive gives the same result
    as reading the record files directly.
    """
    record_name = f"sample-data/multi-segment/{record}"
    archive_path = os.fspath(tmp_path / (os.path.basename(record_name) + ".wfdb"))
    create_archive(multi_record_files(record_name), archive_path)

    for physical in [True, False]:
        try:
            expected = wfdb.rdrecord(record_name, physical=physical, **kwargs)
        except Exception as e:
            # For example, variable layout records can't be converted to
            # a single segment in digital format
            with pytest.raises(type(e)):
                wfdb.rdrecord(archive_path, physical=physical, **kwargs)
            continue
        record = wfdb.rdrecord(archive_path, physical=physical, **kwargs)
        assert record.sig_name == expected.sig_name
        attr = "p_signal" if physical else "d_signal"
        np.testing.assert_array_equal(getattr(record, attr), getattr(expected, attr))

    record = wfdb.rdrecord(archive_path, m2s=False, **kwargs)
    expected = wfdb.rdrecord(record_name, m2s=False, **kwargs)
    assert record.seg_name == expected.seg_name
    assert [s is None for s in record.segments] == [
        s is None for s in expected.segments
    ]


def test_rdheader_multi_segment_archive(tmp_path):
    record_name = "sample-data/multi-segment/s25047/s25047-2704-05-04-10-44"
    archive_path = os.fspath(tmp_path / "s25047.wfdb")
    create_archive(multi_record_files(record_name), archive_path)

    header = wfdb.rdheader(archive_path, rd_segments=True)
    expected = wfdb.rdheader(record_name, rd_segments=True)

    assert header.record_name == expected.record_name
    assert header.seg_name == expected.seg_name
    assert header.sig_name == expected.sig_name
    assert header.sig_segments == expected.sig_segments
    assert [s.sig_name if s else None for s in header.segments] == [
        s.sig_name if s else None for s in expected.segments
    ]


def test_read_multi_segment_archive_missing_segment(tmp_path):
    """
    A segment that the main header lists but that is not in the archive
    is an error.
    """
    record_name = "sample-data/multi-segment/041s/041s"
    archive_path = os.fspath(tmp_path / "041s.wfdb")
    with zipfile.ZipFile(archive_path, "w") as zf:
        for path in multi_record_files(record_name):
            if "041s02" not in path:
                zf.write(path, arcname=os.path.basename(path))

    with pytest.raises(FileNotFoundError, match="041s02.hea"):
        wfdb.rdrecord(archive_path)


def test_multi_segment_archive_with_other_header(tmp_path):
    """
    An archive of a multi-segment record can't contain a header that is
    not one of the record's segments.
    """
    record_dir = "sample-data/multi-segment/s00001"
    record_name = os.path.join(record_dir, "s00001-2896-10-10-00-31")
    archive_path = os.fspath(tmp_path / "s00001.wfdb")

    with pytest.raises(ValueError, match="holds a single record"):
        create_archive(
            multi_record_files(record_name)
            + [os.path.join(record_dir, "3248903_layout.hea")],
            archive_path,
        )

    assert os.listdir(tmp_path) == []


def test_read_archive_unsupported_locations():
    with pytest.raises(ValueError, match="pn_dir"):
        wfdb.rdrecord("100.wfdb", pn_dir="mitdb")
    with pytest.raises(NotImplementedError, match="cloud storage"):
        wfdb.rdrecord("s3://bucket/100.wfdb")


def test_create_archive_compression(tmp_path):
    """
    Headers are compressed, signal files are stored as they are so that
    parts of them can be read without decompressing the whole file.
    """
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    with zipfile.ZipFile(archive_path) as zf:
        compress_types = {info.filename: info.compress_type for info in zf.infolist()}

    assert compress_types == {
        "100.hea": zipfile.ZIP_DEFLATED,
        "100.dat": zipfile.ZIP_STORED,
    }


def test_create_archive_replaces_existing(tmp_path):
    archive_path = os.fspath(tmp_path / "record.wfdb")
    create_archive(record_files("sample-data/test01_00s"), archive_path)

    create_archive(record_files("sample-data/100"), archive_path)

    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == ["100.dat", "100.hea"]


def test_archive_write_error_keeps_existing_archive(tmp_path):
    """
    If an error occurs while writing, the archive at the target path is
    left as it was and no temporary files are left behind.
    """
    archive_path = archive_sample_record("sample-data/100", tmp_path)
    with open(archive_path, "rb") as f:
        original = f.read()

    with pytest.raises(RuntimeError):
        with WFDBArchive(archive_path, mode="w") as archive:
            archive.add_file("sample-data/test01_00s.hea")
            raise RuntimeError("failed while writing")

    with open(archive_path, "rb") as f:
        assert f.read() == original
    assert os.listdir(tmp_path) == ["100.wfdb"]


def test_archive_mode_checks(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    with WFDBArchive(archive_path, mode="r") as archive:
        with pytest.raises(ValueError, match="cannot be written to"):
            archive.write("extra.txt", b"data")

    with WFDBArchive(os.fspath(tmp_path / "new.wfdb"), mode="w") as archive:
        archive.write("new.hea", b"new 0\n")
        # Writing the same content again is allowed, and has no effect
        archive.write("new.hea", b"new 0\n")
        with pytest.raises(ValueError, match="different content"):
            archive.write("new.hea", b"new 1\n")
        with pytest.raises(ValueError, match="cannot be read from"):
            archive.open("new.hea")
    with zipfile.ZipFile(tmp_path / "new.wfdb") as zf:
        assert zf.namelist() == ["new.hea"]

    with pytest.raises(ValueError, match="mode"):
        WFDBArchive(archive_path, mode="a")


def test_archive_holds_one_record(tmp_path):
    """
    An archive holds one record, so that it is clear which record to
    read. A second header that isn't a segment of the record is refused
    when the archive is closed, and the archive is not saved. Other files
    can be added.
    """
    archive_path = os.fspath(tmp_path / "100.wfdb")

    with pytest.raises(ValueError, match="holds a single record"):
        create_archive(
            record_files("sample-data/100") + ["sample-data/test01_00s.hea"],
            archive_path,
        )
    assert os.listdir(tmp_path) == []

    with pytest.raises(ValueError, match="holds a single record"):
        with WFDBArchive(archive_path, mode="w") as archive:
            for path in record_files("sample-data/100"):
                archive.add_file(path)
            archive.write("other.hea", b"other 0\n")
    assert os.listdir(tmp_path) == []

    with WFDBArchive(archive_path, mode="w") as archive:
        for path in record_files("sample-data/100"):
            archive.add_file(path)
        archive.add_file("sample-data/100.atr")

    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == ["100.atr", "100.dat", "100.hea"]


def write_kwargs(fmt="16"):
    rng = np.random.default_rng(1234)
    return dict(
        fs=250,
        units=["mV", "mV"],
        sig_name=["I", "II"],
        p_signal=rng.standard_normal((1000, 2)),
        fmt=[fmt, fmt],
    )


@pytest.mark.parametrize("fmt", ["16", "24", "32", "516"])
def test_wrsamp_archive_matches_files(fmt, tmp_path):
    """
    Writing a record into an archive and reading it back gives the same
    result as writing and reading separate files.
    """
    kwargs = write_kwargs(fmt)
    archive_path = os.fspath(tmp_path / "archive" / "rec.wfdb")
    files_dir = tmp_path / "files"
    files_dir.mkdir()

    wfdb.wrsamp("rec", wfdb_archive=archive_path, **kwargs)
    wfdb.wrsamp("rec", write_dir=os.fspath(files_dir), **kwargs)

    for physical in [True, False]:
        record = wfdb.rdrecord(archive_path, physical=physical)
        expected = wfdb.rdrecord(os.fspath(files_dir / "rec"), physical=physical)
        assert_records_equal(record, expected)


def test_wrsamp_archive_object(tmp_path):
    """
    A WFDBArchive opened by the caller is written to and left open, and no
    separate files are written.
    """
    archive_path = os.fspath(tmp_path / "rec.wfdb")

    with WFDBArchive(archive_path, mode="w") as archive:
        wfdb.wrsamp(
            "rec",
            write_dir=os.fspath(tmp_path),
            wfdb_archive=archive,
            **write_kwargs(),
        )
        archive.write("notes.txt", b"extra file")

    assert os.listdir(tmp_path) == ["rec.wfdb"]
    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == ["notes.txt", "rec.dat", "rec.hea"]
    assert wfdb.rdrecord(archive_path).sig_len == 1000


def test_record_wrsamp_archive_round_trip(tmp_path):
    """
    A record read from an archive can be written to a new archive.
    """
    source = archive_sample_record("sample-data/100", tmp_path)
    record = wfdb.rdrecord(source, physical=False)
    copy_path = os.fspath(tmp_path / "copy" / "100.wfdb")

    record.wrsamp(wfdb_archive=copy_path)

    assert_records_equal(
        wfdb.rdrecord(copy_path, physical=False),
        wfdb.rdrecord("sample-data/100", physical=False),
    )


def test_wrsamp_archive_replaces_existing(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    wfdb.wrsamp("100", wfdb_archive=archive_path, **write_kwargs())

    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == ["100.dat", "100.hea"]
    assert wfdb.rdrecord(archive_path).sig_name == ["I", "II"]


def test_wrsamp_archive_error_writes_nothing(tmp_path, monkeypatch):
    """
    If writing the record fails after the header was written, no archive
    or temporary file is left.
    """

    def fail(*args, **kwargs):
        raise RuntimeError("failed while writing the dat file")

    monkeypatch.setattr(wfdb.io._signal, "wr_dat_file", fail)

    with pytest.raises(RuntimeError, match="dat file"):
        wfdb.wrsamp(
            "rec",
            wfdb_archive=os.fspath(tmp_path / "rec.wfdb"),
            **write_kwargs(),
        )

    assert os.listdir(tmp_path) == []


def test_wrsamp_two_records_in_one_archive(tmp_path):
    """
    Writing a second record into an archive fails, and the archive is not
    saved.
    """
    archive_path = os.fspath(tmp_path / "rec.wfdb")

    with pytest.raises(ValueError, match="holds a single record"):
        with WFDBArchive(archive_path, mode="w") as archive:
            wfdb.wrsamp("rec", wfdb_archive=archive, **write_kwargs())
            wfdb.wrsamp("other", wfdb_archive=archive, **write_kwargs())

    assert os.listdir(tmp_path) == []


def test_wrsamp_archive_invalid_argument(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    with pytest.raises(TypeError, match="wfdb_archive"):
        wfdb.wrsamp("rec", wfdb_archive=123, **write_kwargs())

    with WFDBArchive(archive_path, mode="r") as archive:
        with pytest.raises(ValueError, match="cannot be written to"):
            wfdb.wrsamp("rec", wfdb_archive=archive, **write_kwargs())


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
def test_archive_permissions(tmp_path):
    """
    A new archive gets the same permissions as other new files, and a
    replaced archive keeps its permissions.
    """
    archive_path = tmp_path / "rec.wfdb"
    other_path = tmp_path / "other.txt"
    other_path.write_text("")

    wfdb.wrsamp("rec", wfdb_archive=os.fspath(archive_path), **write_kwargs())
    assert archive_path.stat().st_mode == other_path.stat().st_mode

    archive_path.chmod(0o640)
    wfdb.wrsamp("rec", wfdb_archive=os.fspath(archive_path), **write_kwargs())
    assert archive_path.stat().st_mode & 0o777 == 0o640


def test_archive_temporary_file(tmp_path):
    """
    While an archive is written, its temporary file is named after it.
    If the archive is discarded without being closed, the temporary file
    is removed and nothing is saved.
    """
    archive = WFDBArchive(os.fspath(tmp_path / "rec.wfdb"), mode="w")
    archive.write("rec.hea", b"rec 0\n")
    (tmp_file,) = os.listdir(tmp_path)
    assert tmp_file.startswith(".rec.wfdb.") and tmp_file.endswith(".tmp")

    del archive
    gc.collect()

    assert os.listdir(tmp_path) == []


def test_rdrecord_opens_archive_once(tmp_path, monkeypatch):
    """
    The header and the signals are read from the same open archive.
    """
    archive_path = archive_sample_record("sample-data/100", tmp_path)
    opened = []
    original_init = WFDBArchive.__init__

    def counting_init(self, *args, **kwargs):
        opened.append(args)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(WFDBArchive, "__init__", counting_init)

    wfdb.rdrecord(archive_path, sampto=100)

    assert len(opened) == 1


def test_rdann_archive(tmp_path):
    """
    Annotations read from an archive match the annotation file on disk.
    """
    archive_path = os.fspath(tmp_path / "100.wfdb")
    create_archive(
        record_files("sample-data/100") + ["sample-data/100.atr"],
        archive_path,
    )

    ann = wfdb.rdann(archive_path, "atr", sampfrom=1000, sampto=50000)
    expected = wfdb.rdann("sample-data/100", "atr", sampfrom=1000, sampto=50000)

    assert ann.record_name == "100"
    assert ann.extension == "atr"
    assert ann.fs == expected.fs
    np.testing.assert_array_equal(ann.sample, expected.sample)
    assert ann.symbol == expected.symbol
    assert ann.aux_note == expected.aux_note


def test_rdann_archive_header_named_differently(tmp_path):
    """
    The annotation file is named after the record's header, which can
    differ from the archive name.
    """
    archive_path = os.fspath(tmp_path / "renamed.wfdb")
    create_archive(
        record_files("sample-data/100") + ["sample-data/100.atr"],
        archive_path,
    )

    ann = wfdb.rdann(archive_path, "atr")

    assert ann.record_name == "100"
    assert len(ann.sample) == len(wfdb.rdann("sample-data/100", "atr").sample)


def test_rdann_archive_errors(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    with pytest.raises(FileNotFoundError, match="100.atr"):
        wfdb.rdann(archive_path, "atr")
    with pytest.raises(ValueError, match="pn_dir"):
        wfdb.rdann(archive_path, "atr", pn_dir="mitdb")


@pytest.mark.parametrize(
    "record, kwargs",
    [
        ("fixed1/v102s", {}),
        ("041s/041s", {}),
        ("s25047/s25047-2704-05-04-10-44", {}),
        (
            "s00001/s00001-2896-10-10-00-31",
            dict(sampfrom=14428364, sampto=14428375),
        ),
    ],
)
def test_multi_record_wrsamp_archive(record, kwargs, tmp_path):
    """
    A multi-segment record written into an archive reads back the same as
    the original record files. In 'fixed1', all segments share one dat
    file, which is written to the archive once. 's25047' and 's00001' have
    a variable layout, with a layout header and empty segments.
    """
    record_name = f"sample-data/multi-segment/{record}"
    archive_path = os.fspath(tmp_path / "multi.wfdb")
    multi_record = wfdb.rdrecord(record_name, m2s=False, physical=False)

    multi_record.wrsamp(wfdb_archive=archive_path)

    assert os.listdir(tmp_path) == ["multi.wfdb"]
    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == sorted(
            os.path.basename(path) for path in multi_record_files(record_name)
        )
    record = wfdb.rdrecord(archive_path, **kwargs)
    expected = wfdb.rdrecord(record_name, **kwargs)
    assert record.sig_name == expected.sig_name
    np.testing.assert_array_equal(record.p_signal, expected.p_signal)
