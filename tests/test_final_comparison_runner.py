"""Fold/seed expansion, explicit candidate selection and final comparison execution."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_experiments as runner


class FinalComparisonRunnerTests(unittest.TestCase):
    def config(self, root, **updates):
        config = json.loads((runner.ROOT/'experiments/final_comparison.json').read_text())
        config['initial_comparison_config'] = str(runner.ROOT/'experiments/initial_comparison.json')
        config.update(updates)
        path = root/'final_comparison.json'
        path.write_text(json.dumps(config))
        return path

    def args(self, root, config, index=0, **updates):
        values = dict(config=config, run_index=index, run_id=None, data_root=root/'data',
                      source_dir=root/'raw', results_root=root/'results', python=Path('/python'),
                      workers=4, processes=4, attempt=1, dry_run=False, prepare_only=False)
        values.update(updates)
        return SimpleNamespace(**values)

    def test_exact_27_run_mapping_and_original_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = runner.load_config(self.config(root, top_candidates={'TOP1':'M1', 'TOP2':'M7'}))
            runs = config['runs']
            self.assertEqual(len(runs), 27)
            self.assertEqual(len({(r['configuration']['id'], r['fold'], r['training_seed']) for r in runs}), 27)
            self.assertEqual([runs[i]['fold'] for i in (0, 3, 6)], [1, 2, 3])
            self.assertEqual([runs[i]['training_seed'] for i in range(3)], [0, 1, 2])
            baseline = runs[0]['configuration']
            self.assertEqual((baseline['slices'], baseline['loss'], baseline['class_weighting']), (1, 'ce', 'none'))
            self.assertFalse(baseline['hu_clipping'] or baseline['tiling'] or baseline['augmentation'])
            self.assertTrue(baseline['original_grid_validation'])
            self.assertEqual(runs[9]['configuration']['source_id'], 'M1')
            self.assertEqual(runs[18]['configuration']['epochs'], 3)

    def test_placeholders_block_and_dry_runs_do_not_write(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner.subprocess, 'run') as process:
            root = Path(d)
            config = self.config(root)
            for index in (9, 18):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Select TOP'):
                    runner.run(self.args(root, config, index))
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    runner.run(self.args(root, config, index, dry_run=True))
                self.assertIn('blocked', output.getvalue())
            self.assertFalse((root/'results').exists())
            process.assert_not_called()

    def test_paths_separate_folds_but_share_preprocessing_between_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = self.config(root)
            plans = []
            for index in (0, 1, 3):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    runner.run(self.args(root, config, index, dry_run=True))
                plans.append(json.loads(output.getvalue()))
            self.assertEqual(plans[0]['dataset'], plans[1]['dataset'])
            self.assertNotEqual(plans[0]['dataset'], plans[2]['dataset'])
            self.assertNotEqual(plans[0]['run_directory'], plans[1]['run_directory'])
            self.assertIn('--original-grid-validation', plans[0]['command'])
            self.assertIn('none', plans[0]['command'])
            self.assertNotIn('--tiling', plans[0]['evaluation_command'])
            self.assertEqual(plans[2]['fold'], 2)

    def test_invalid_or_duplicate_selections_and_folds(self):
        for update in [{'top_candidates':{'TOP1':'M1','TOP2':'M1'}},
                       {'top_candidates':{'TOP1':'missing','TOP2':None}},
                       {'folds':[1,1,2]}, {'training_seeds':[-1,0,1]}]:
            with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
                runner.load_config(self.config(Path(d), **update))

    def test_prepare_only_never_launches_training(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(runner, 'prepare_dataset') as prepare, \
                patch.object(runner.subprocess, 'run') as process:
            root = Path(d)
            config = self.config(root, top_candidates={'TOP1':'M7','TOP2':None})
            runner.run(self.args(root, config, 9, prepare_only=True))
            prepare.assert_called_once()
            command = prepare.call_args.args[-1]
            self.assertIn('--tiling', command)
            self.assertIn('--hu_clip', command)
            self.assertIn('--save_metadata', command)
            self.assertEqual(command[command.index('--fold')+1], '1')
            process.assert_not_called()

    def test_final_evaluation_required_before_completed_status(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(runner, 'check_dataset', return_value={}), \
                patch.object(runner, 'summarize', return_value={'best_epoch':0, 'mean_foreground_dice':.5,
                                                             'evaluation_grid':'original_CT'}):
            root = Path(d)
            config = self.config(root)
            args = self.args(root, config)
            def execute(command, **kwargs):
                if '--evaluate-checkpoint' in command:
                    folder = Path(command[command.index('--dest')+1])
                    folder.mkdir()
                    (folder/'metrics.json').write_text(json.dumps({'grid':'original_CT', 'mean_foreground_dice':.5}))
            with patch.object(runner.subprocess, 'run', side_effect=execute) as process:
                # Filling unrelated candidate placeholders must not invalidate baseline runs.
                self.config(root, top_candidates={'TOP1':'M1', 'TOP2':'M7'})
                runner.run(args)
                self.assertEqual(process.call_count, 3)
                manifest_path = root/'results/final_comparison/BASELINE/fold_1/seed_0/attempt_001/runner.json'
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest['status'], 'completed')
                self.assertEqual(manifest['metrics']['evaluation_grid'], 'original_CT')
                runner.run(args)
                self.assertEqual(process.call_count, 3)


if __name__ == '__main__':
    unittest.main()
