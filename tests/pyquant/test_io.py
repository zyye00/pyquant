import pandas as pd
import pytest

from pyquant import ensure_dir, save_output


def test_save_output_does_not_overwrite_by_default(tmp_path):
    path = tmp_path / "out.csv"
    save_output(pd.DataFrame({"a": [1]}), path)

    with pytest.raises(FileExistsError):
        save_output(pd.DataFrame({"a": [2]}), path)


def test_ensure_dir(tmp_path):
    path = ensure_dir(tmp_path / "a" / "b")

    assert path.exists()
