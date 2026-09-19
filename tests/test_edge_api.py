from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecosort_edge.server import EdgeApiServer
from ecosort_edge.state import EdgeStateStore


class EdgeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = EdgeStateStore(monitored_bin="plastic", empty_depth_cm=30)
        self.state.update_distance_mm(80)
        self.state.record_disposal(route="plastic", confidence=0.94, detected_object="PET Bottle")
        self.server = EdgeApiServer(self.state, host="127.0.0.1", port=0).start()
        self.base_url = f"http://127.0.0.1:{self.server.port}"

    def tearDown(self) -> None:
        self.server.stop()

    def get_json(self, path: str) -> dict[str, object]:
        with urlopen(f"{self.base_url}{path}", timeout=2) as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
            return json.load(response)

    def test_status_bins_and_events(self) -> None:
        status = self.get_json("/api/status")
        self.assertTrue(status["online"])
        self.assertEqual(status["monitored_bin"], "plastic")

        bins = self.get_json("/api/bins")["bins"]
        plastic = next(item for item in bins if item["id"] == "plastic")
        self.assertEqual(plastic["fill_state"], "half-full")

        events = self.get_json("/api/events")["events"]
        self.assertEqual(events[0]["object"], "PET Bottle")

    def test_mark_emptied_endpoint(self) -> None:
        request = Request(
            f"{self.base_url}/api/bins/plastic/emptied",
            data=b"",
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["bin"]["fill_state"], "empty")
        self.assertEqual(payload["event"]["kind"], "maintenance")

    def test_webcam_fill_state_endpoint(self) -> None:
        request = Request(
            f"{self.base_url}/api/bins/plastic/fill-state",
            data=json.dumps(
                {"fill_state": "full", "source": "webcam", "confidence": 0.96}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["bin"]["fill_state"], "full")
        self.assertEqual(payload["bin"]["fill_source"], "webcam")

    def test_unknown_path_is_404(self) -> None:
        with self.assertRaises(HTTPError) as caught:
            self.get_json("/api/missing")
        self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
