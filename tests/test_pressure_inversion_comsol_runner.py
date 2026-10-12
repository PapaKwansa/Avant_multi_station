"""Input-contract tests for the exploratory observed-COMSOL runner."""
from pathlib import Path
import importlib.util
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'inversion' / 'run_pressure_inversion_comsol.py'
spec = importlib.util.spec_from_file_location('observed_comsol_runner', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_time_selection():
    assert module.choose_times(8, [5, 2], False).tolist() == [2, 5]
    assert module.choose_times(5, None, True).tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize('indices', [[0], [3, 3], [10], [-1], [True], []])
def test_invalid_time_selection(indices):
    with pytest.raises(ValueError):
        module.choose_times(8, indices, False)


def test_holdout_channels():
    train, hold = module.channel_masks(['eXX_S01', 'eXX_S02', 'eYY_S01', 'eYY_S02'], ['S02'])
    assert train.tolist() == [True, False, True, False]
    assert hold.tolist() == [False, True, False, True]


@pytest.mark.parametrize('holdouts', [['S03'], ['S01', 'S02'], ['S01', 'S01'], []])
def test_invalid_holdouts(holdouts):
    with pytest.raises(ValueError):
        module.channel_masks(['eXX_S01', 'eXX_S02'], holdouts)


def test_coarse_divisibility_and_noise():
    cfg = dict(fine_splits=[4,4,1], pressure_models={'uniform':[1,1,1], 'four_zone':[2,2,1]},
               prior_sd_pa=1e6, prior_mean_pa=0, assumed_error_std_ns=2, reference_mode='first_sample')
    assert module.validate_settings(cfg, {'station_file':'converted.csv'}) == (4,4,1)
    cfg['pressure_models']['invalid'] = [3,2,1]
    with pytest.raises(ValueError):
        module.validate_settings(cfg, {'station_file':'converted.csv'})
