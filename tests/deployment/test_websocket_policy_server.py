import pytest

pytest.importorskip("websockets", reason="websockets is required (see deployment/DEPLOYMENT.md)")
pytest.importorskip("msgpack", reason="msgpack is required (see deployment/DEPLOYMENT.md)")

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls = []
        self.reset_count = 0

    def predict_action(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        return {"actions": [[0.0]]}

    def reset(self) -> None:
        self.reset_count += 1


@pytest.mark.parametrize("msg", [["not", "a", "dict"], "infer", None, b"raw", 7])
def test_non_dict_message_returns_error_instead_of_raising(msg: object) -> None:
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(policy)

    response = server._route_message(msg)

    assert response == {
        "status": "error",
        "ok": False,
        "type": "inference_result",
        "request_id": "default",
        "error": {"message": "Payload must be a dict", "payload_type": str(type(msg))},
    }
    assert policy.calls == []


def test_flat_infer_message_is_wrapped_as_one_example() -> None:
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(policy)

    message = {"type": "infer", "request_id": "r1", "state": [1.0], "prompt": "task"}

    response = server._route_message(message)

    assert policy.calls == [{"examples": [{"state": [1.0], "prompt": "task"}]}]
    assert response == {
        "status": "ok",
        "ok": True,
        "type": "inference_result",
        "request_id": "r1",
        "actions": [[0.0]],
    }


def test_message_without_type_is_an_infer_request() -> None:
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(policy)

    response = server._route_message({"state": [1.0]})

    assert policy.calls == [{"examples": [{"state": [1.0]}]}]
    assert response["request_id"] == "default"
    assert response["ok"] is True


def test_ping_reset_and_unknown_types_still_route() -> None:
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(policy)

    assert server._route_message({"type": "ping", "request_id": "p"}) == {
        "status": "ok",
        "ok": True,
        "type": "ping",
        "request_id": "p",
    }
    assert server._route_message({"type": "reset"})["ok"] is True
    assert policy.reset_count == 1
    unknown = server._route_message({"type": "bogus"})
    assert unknown["ok"] is False
    assert unknown["error"] == {"message": "Unsupported message type 'bogus'"}
