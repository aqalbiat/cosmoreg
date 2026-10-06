import numpy as np

from camels_pipeline.config import PipelineConfig
from camels_pipeline.splits import make_splits, split_summary


def test_default_split_sizes_and_complete_coverage():
    config = PipelineConfig()
    splits = make_splits(1000, config)
    summary = split_summary(splits, config.maps_per_simulation)

    assert sum(group["simulations"] for group in summary["level1"].values()) == 67
    assert sum(group["maps"] for group in summary["level1"].values()) == 1005
    assert summary["level1"]["test"]["simulations"] == 13
    assert [summary["level1"][f"fold{i}"]["simulations"] for i in range(1, 6)] == [11, 11, 11, 11, 10]

    assert sum(group["maps"] for group in summary["level2"].values()) == 4995
    assert summary["level2"]["test"]["simulations"] == 67
    assert [summary["level2"][f"fold{i}"]["simulations"] for i in range(1, 6)] == [54, 53, 53, 53, 53]

    assert sum(group["maps"] for group in summary["level3"].values()) == 9000
    assert summary["level3"]["test"]["simulations"] == 120
    assert [summary["level3"][f"fold{i}"]["simulations"] for i in range(1, 6)] == [96, 96, 96, 96, 96]

    all_ids = np.concatenate([ids for groups in splits.values() for ids in groups.values()])
    assert len(all_ids) == 1000
    assert len(np.unique(all_ids)) == 1000


def test_splits_are_reproducible_and_seeded():
    first = make_splits(1000, PipelineConfig(seed=123))
    second = make_splits(1000, PipelineConfig(seed=123))
    third = make_splits(1000, PipelineConfig(seed=124))
    assert all(np.array_equal(first[level][group], second[level][group]) for level in first for group in first[level])
    assert any(not np.array_equal(first[level][group], third[level][group]) for level in first for group in first[level])
