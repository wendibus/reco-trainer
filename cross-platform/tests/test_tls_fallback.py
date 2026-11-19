import ssl
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import ml_worker


def cert_error() -> urllib.error.URLError:
    return urllib.error.URLError(ssl.SSLCertVerificationError("unable to get issuer certificate"))


class TlsFallbackTests(unittest.TestCase):
    def test_retries_with_next_store_after_certificate_error(self) -> None:
        contexts = [ssl.create_default_context(), ssl.create_default_context()]
        seen: list[ssl.SSLContext] = []

        def fake_urlopen(request, timeout, context):
            seen.append(context)
            if context is contexts[0]:
                raise cert_error()
            return "response"

        with mock.patch.object(ml_worker, "tls_contexts", return_value=contexts), mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            self.assertEqual(ml_worker.open_verified_url(urllib.request.Request("https://example.org")), "response")
        self.assertEqual(seen, contexts)

    def test_raises_when_every_store_fails(self) -> None:
        with mock.patch.object(ml_worker, "tls_contexts", return_value=[ssl.create_default_context()] * 2), mock.patch.object(urllib.request, "urlopen", side_effect=cert_error()):
            with self.assertRaises(urllib.error.URLError):
                ml_worker.open_verified_url(urllib.request.Request("https://example.org"))

    def test_other_errors_are_not_retried(self) -> None:
        calls = []

        def fake_urlopen(request, timeout, context):
            calls.append(1)
            raise urllib.error.URLError(OSError("offline"))

        with mock.patch.object(ml_worker, "tls_contexts", return_value=[ssl.create_default_context()] * 2), mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            with self.assertRaises(urllib.error.URLError):
                ml_worker.open_verified_url(urllib.request.Request("https://example.org"))
        self.assertEqual(len(calls), 1)

    def test_certificate_checks_stay_on(self) -> None:
        for context in ml_worker.tls_contexts():
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)


if __name__ == "__main__":
    unittest.main()
