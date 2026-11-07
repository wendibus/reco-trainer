import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import local_worker
import ml_worker


class UserErrorTests(unittest.TestCase):
    def test_a_message_exit_becomes_a_user_error_line_and_code_3(self):
        def stop(_args):
            raise SystemExit("No flagged images.\nRun “Check consistency” first.")
        parser = mock.Mock(parse_args=lambda: mock.Mock(func=stop))
        output = io.StringIO()
        with mock.patch.object(ml_worker, "build_parser", return_value=parser), redirect_stdout(output), \
                mock.patch.object(ml_worker.signal, "signal"):
            with self.assertRaises(SystemExit) as caught:
                ml_worker.main()
        self.assertEqual(caught.exception.code, 3)
        self.assertEqual(output.getvalue().strip(), "RECO_USER_ERROR No flagged images. Run “Check consistency” first.")

    def test_a_crash_keeps_its_exit_code(self):
        def stop(_args):
            raise SystemExit(1)
        parser = mock.Mock(parse_args=lambda: mock.Mock(func=stop))
        with mock.patch.object(ml_worker, "build_parser", return_value=parser), mock.patch.object(ml_worker.signal, "signal"):
            with self.assertRaises(SystemExit) as caught:
                ml_worker.main()
        self.assertEqual(caught.exception.code, 1)

    def test_runner_raises_the_message_and_skips_the_gpu_diagnosis(self):
        process = mock.Mock(stdout=io.StringIO("loading\nRECO_USER_ERROR No flagged images.\n"), wait=lambda: 3)
        with mock.patch.object(local_worker, "popen_tracked", return_value=process), mock.patch.object(local_worker, "untrack"), \
                mock.patch.object(local_worker, "STATE", local_worker.LocalState()):
            with self.assertRaises(local_worker.WorkerUserError) as caught:
                local_worker.run_logged(["python", "ml_worker.py"])
            self.assertEqual(str(caught.exception), "No flagged images.")
            with mock.patch.object(local_worker, "amd_choice", return_value="yes"), \
                    mock.patch.object(local_worker, "record_gpu_problem") as problem:
                local_worker.record_job_failure("consistency-suggest", "No flagged images.")
                problem.assert_not_called()
                local_worker.record_job_failure("consistency-suggest", "The local ML worker exited with code 1.")
                problem.assert_called_once()


class SuggestFramesTests(unittest.TestCase):
    def test_suggestions_cover_the_frames_the_app_lists(self):
        frames = [
            {"id": "a", "reviewFlags": ["detector-missing"], "heldOut": True, "reviewStatus": "reviewed"},
            {"id": "b", "reviewFlags": ["detector-missing"], "reviewStatus": "candidate"},
            {"id": "c", "reviewFlags": []},
        ]

        class Selected(Exception):
            pass
        args = mock.Mock(project=".", language="en", tier=None, teacher=None, iou=0.5)
        with mock.patch.object(ml_worker, "require_project", return_value={"frames": frames}), \
                mock.patch.object(ml_worker, "hardware_tier"), mock.patch.object(ml_worker, "teacher_size", return_value="nano"), \
                mock.patch.object(ml_worker, "emit") as emit, mock.patch.object(ml_worker, "model_instance", side_effect=Selected):
            with self.assertRaises(Selected):
                ml_worker.consistency_suggest(args)
        self.assertIn("looks at 1 flagged", emit.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
