import pytest
import tempfile
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock
from uec.cli import run_battery, run_aot


class TestRunBattery:
    def test_run_battery_basic(self, capsys):
        # Test basic battery run with fast mode
        with patch('uec.cli.random_markov_biased') as mock_markov, \
             patch('uec.cli.sample_markov') as mock_sample, \
             patch('uec.cli.entropy_production_rate_bits') as mock_ep, \
             patch('uec.cli.klrate_holonomy_time_reversal_markov') as mock_hol:
            
            mock_markov.return_value = [[0.7, 0.3], [0.4, 0.6]]
            mock_sample.return_value = [0, 1, 0, 1, 0] * 100
            mock_ep.return_value = 0.1
            mock_hol.return_value = 0.09
            
            run_battery(['--fast', '--seed', '123', '--n', '1000'])
            
            captured = capsys.readouterr()
            assert 'UEC Battery (minimal): starting' in captured.out
            assert 'UEC Battery (minimal): done' in captured.out
            assert 'EP analytic' in captured.out
            assert 'KL-rate hol' in captured.out
            
    def test_run_battery_with_parameters(self, capsys):
        # Test battery with custom parameters
        with patch('uec.cli.random_markov_biased') as mock_markov, \
             patch('uec.cli.sample_markov') as mock_sample, \
             patch('uec.cli.entropy_production_rate_bits') as mock_ep, \
             patch('uec.cli.klrate_holonomy_time_reversal_markov') as mock_hol:
            
            mock_markov.return_value = [[0.8, 0.2], [0.3, 0.7]]
            mock_sample.return_value = [0, 1] * 250
            mock_ep.return_value = 0.05
            mock_hol.return_value = 0.048
            
            run_battery(['--seed', '456', '--n', '500', '--k', '2', '--order', '2'])
            
            captured = capsys.readouterr()
            assert 'UEC Battery (minimal): starting' in captured.out
            assert 'UEC Battery (minimal): done' in captured.out
            
            # Check that functions were called with correct parameters
            mock_markov.assert_called_once()
            mock_sample.assert_called_once()
            mock_ep.assert_called_once()
            mock_hol.assert_called_once()
            
            # Check k=2 and R=2 were used
            args, kwargs = mock_hol.call_args
            assert kwargs['k'] == 2
            assert kwargs['R'] == 2


class TestRunAoT:
    def test_run_aot_csv_basic(self, capsys):
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,value\n")
            f.write("1,1.5\n")
            f.write("2,2.7\n") 
            f.write("3,1.2\n")
            f.write("4,3.1\n")
            temp_path = f.name
            
        try:
            with patch('uec.cli.aot_from_series') as mock_aot:
                mock_aot.return_value = {
                    'auc': 0.75,
                    'bits_per_step': 0.12,
                    'hol_ci_lo': 0.10,
                    'hol_ci_hi': 0.14
                }
                
                run_aot(['--aot_csv', temp_path, '--aot_csv_col', 'value'])
                
                captured = capsys.readouterr()
                assert '[AoT CSV]' in captured.out
                assert 'AUC=0.750' in captured.out
                assert 'bits/step=0.12' in captured.out
                
                # Check JSON output
                lines = captured.out.strip().split('\n')
                json_line = [line for line in lines if line.startswith('{"file"')][0]
                result = json.loads(json_line)
                assert result['file'] == temp_path
                assert result['auc'] == 0.75
                
        finally:
            os.unlink(temp_path)
            
    def test_run_aot_csv_by_index(self, capsys):
        # Test CSV loading by column index
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write("1.0,2.0,3.0\n")
            f.write("4.0,5.0,6.0\n")
            temp_path = f.name
            
        try:
            with patch('uec.cli.aot_from_series') as mock_aot:
                mock_aot.return_value = {
                    'auc': 0.60,
                    'bits_per_step': 0.08,
                    'hol_ci_lo': 0.06,
                    'hol_ci_hi': 0.10
                }
                
                run_aot(['--aot_csv', temp_path, '--aot_csv_col', '1'])  # second column
                
                captured = capsys.readouterr()
                assert '[AoT CSV]' in captured.out
                
                # Check that aot_from_series was called
                mock_aot.assert_called_once()
                
        finally:
            os.unlink(temp_path)
            
    def test_run_aot_wav(self, capsys):
        # Test WAV file processing using data/wav files
        from pathlib import Path
        wav_path = Path(__file__).parent.parent / "data" / "wav" / "427624__polaina_legal__sine.wav"
        
        if not wav_path.exists():
            pytest.skip("Test WAV file not found")
            
        with patch('uec.cli.aot_from_series') as mock_aot:
            mock_aot.return_value = {
                'auc': 0.68,
                'bits_per_step': 0.09,
                'bits_per_second': 4320.5,
                'hol_ci_lo': 0.07,
                'hol_ci_hi': 0.11
            }
            
            run_aot(['--aot_wav', str(wav_path)])
            
            captured = capsys.readouterr()
            assert '[AoT WAV]' in captured.out
            assert 'AUC=0.680' in captured.out
            assert 'bits/step=0.09' in captured.out
            assert 'bits/s=4320.5' in captured.out
            assert 'Hz' in captured.out
            
            # Check that aot_from_series was called with correct parameters
            mock_aot.assert_called_once()
            args, kwargs = mock_aot.call_args
            assert len(args[0]) > 0  # Audio data should be non-empty
            assert kwargs.get('use_logreturn') == False  # WAV doesn't use log returns
        
    def test_run_aot_scoreboard_mode(self, capsys):
        # Create temporary CSV files for scoreboard
        files = []
        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    f.write("value\n")
                    f.write(f"{i+1}.5\n")
                    f.write(f"{i+2}.7\n")
                    f.write(f"{i+3}.2\n")
                    f.write(f"{i+4}.1\n")
                    f.write(f"{i+5}.8\n")
                    files.append(f.name)
                    
            with patch('uec.cli.aot_from_series') as mock_aot, \
                 patch('uec.cli.load_csv_column') as mock_load_csv, \
                 patch('glob.glob') as mock_glob:
                
                mock_glob.return_value = files
                mock_load_csv.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
                mock_aot.return_value = {
                    'auc': 0.65,
                    'bits_per_step': 0.10,
                    'bits_per_second': None,
                    'hol_ci_lo': 0.08,
                    'hol_ci_hi': 0.12
                }
                
                run_aot(['--scoreboard_glob', '*.csv'])
                
                captured = capsys.readouterr()
                assert '[Scoreboard]' in captured.out
                
                # Should process multiple files
                assert mock_aot.call_count == len(files)
                
        finally:
            for file_path in files:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    
    def test_run_aot_with_preprocessing_flags(self, capsys):
        # Test with different preprocessing options
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("price\n")
            f.write("100.0\n")
            f.write("102.0\n")
            f.write("98.0\n")
            temp_path = f.name
            
        try:
            with patch('uec.cli.aot_from_series') as mock_aot:
                mock_aot.return_value = {
                    'auc': 0.55,
                    'bits_per_step': 0.05,
                    'hol_ci_lo': 0.03,
                    'hol_ci_hi': 0.07
                }
                
                run_aot([
                    '--aot_csv', temp_path,
                    '--aot_csv_col', 'price',
                    '--aot_diff',
                    '--aot_logreturn',
                    '--aot_bins', '4',
                    '--aot_win', '128',
                    '--aot_stride', '64',
                    '--aot_rate', '1000',
                    '--order', '2'
                ])
                
                captured = capsys.readouterr()
                assert '[AoT CSV]' in captured.out
                
                # Check that aot_from_series was called with correct parameters
                args, kwargs = mock_aot.call_args
                assert kwargs['k'] == 4
                assert kwargs['R'] == 2
                assert kwargs['win'] == 128
                assert kwargs['stride'] == 64
                assert kwargs['use_diff'] == True
                assert kwargs['use_logreturn'] == True
                assert kwargs['sr'] == 1000
                
        finally:
            os.unlink(temp_path)
            
    def test_run_aot_no_args_shows_help(self, capsys):
        # Test that running with no arguments shows help
        with patch('sys.stdout') as mock_stdout:
            run_aot([])
            # Help should be shown (argparse calls sys.stdout)
            mock_stdout.write.assert_called()
            
    def test_run_aot_with_seed(self, capsys):
        # Test reproducible results with seed
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("data\n")
            f.write("1.0\n")
            f.write("2.0\n")
            temp_path = f.name
            
        try:
            with patch('uec.cli.aot_from_series') as mock_aot:
                mock_aot.return_value = {
                    'auc': 0.70,
                    'bits_per_step': 0.15,
                    'hol_ci_lo': 0.12,
                    'hol_ci_hi': 0.18
                }
                
                run_aot([
                    '--aot_csv', temp_path,
                    '--seed', '42'
                ])
                
                # Check that RNG was passed to aot_from_series
                args, kwargs = mock_aot.call_args
                assert 'rng' in kwargs
                assert kwargs['rng'] is not None
                
        finally:
            os.unlink(temp_path)


def test_cli_functions_integration():
    """Test that CLI functions can be imported and called without errors."""
    # Test that functions exist and are callable
    assert callable(run_battery)
    assert callable(run_aot)
    
    # Test basic parameter validation
    with pytest.raises(SystemExit):  # argparse exits on invalid args
        run_battery(['--invalid_argument'])
        
    with pytest.raises(SystemExit):
        run_aot(['--invalid_argument'])