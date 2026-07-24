"""Tests for the streaming module."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.fermax_blue.streaming import (
    ConsumeResult,
    FermaxSignalingClient,
    FermaxStreamSession,
    RoomJoinResult,
    TransportData,
    streaming_deps_available,
)


class TestStreamingDepsAvailable:
    """Detection of the optional live-video dependencies."""

    def test_true_when_pymediasoup_installed(self):
        streaming_deps_available.cache_clear()
        assert streaming_deps_available() is True
        streaming_deps_available.cache_clear()

    def test_false_when_pymediasoup_missing(self):
        streaming_deps_available.cache_clear()
        with patch(
            "custom_components.fermax_blue.streaming.find_spec",
            return_value=None,
        ):
            assert streaming_deps_available() is False
        streaming_deps_available.cache_clear()


def _mock_transport() -> MagicMock:
    transport = MagicMock()
    transport.on = MagicMock(return_value=lambda f: f)
    transport.close = AsyncMock()
    return transport


def _mocked_session(tmp_path, receive_only):
    """Build a session with the signaling client and mediasoup device mocked out."""
    session = FermaxStreamSession(
        signaling_url="https://signaling-pro-duoxme.fermax.io",
        oauth_token="oauth",
        fcm_token="fcm",
        room_id="room1",
        media_root=str(tmp_path),
        receive_only=receive_only,
    )

    transport_data = TransportData(
        id="t1", dtls_parameters="{}", ice_candidates="[]", ice_parameters="{}"
    )
    signaling = MagicMock()
    signaling.connect = AsyncMock(
        return_value=RoomJoinResult(
            video_producer_id="vp",
            audio_producer_id="ap",
            router_rtp_capabilities="{}",
            recv_video_transport=transport_data,
            recv_audio_transport=transport_data,
            send_transport=transport_data,
        )
    )
    signaling.consume_transport = AsyncMock(
        return_value=ConsumeResult(
            consumer_id="c1", producer_id="vp", kind="video", rtp_parameters=MagicMock()
        )
    )
    signaling.connect_transport = AsyncMock(return_value=True)
    signaling.pickup = AsyncMock()
    signaling.disconnect = AsyncMock()
    session._signaling = signaling

    from aiortc.mediastreams import MediaStreamError

    consumer = MagicMock()
    consumer.track.recv = AsyncMock(side_effect=MediaStreamError)
    consumer.close = AsyncMock()
    recv_transport = _mock_transport()
    recv_transport.consume = AsyncMock(return_value=consumer)

    producer = MagicMock()
    producer.close = AsyncMock()
    send_transport = _mock_transport()
    send_transport.produce = AsyncMock(return_value=producer)

    device = MagicMock()
    device.load = AsyncMock()
    device.createRecvTransport = MagicMock(return_value=recv_transport)
    device.createSendTransport = MagicMock(return_value=send_transport)
    device.rtpCapabilities.dict = MagicMock(return_value={})

    patches = (
        patch("pymediasoup.Device", return_value=device),
        patch("pymediasoup.rtp_parameters.RtpCapabilities", MagicMock()),
        patch("pymediasoup.models.transport.DtlsParameters", MagicMock()),
        patch("pymediasoup.models.transport.IceCandidate", MagicMock()),
        patch("pymediasoup.models.transport.IceParameters", MagicMock()),
    )
    return session, send_transport, patches


class TestReceiveOnlySession:
    """Receive-only sessions consume video but never signal pickup."""

    async def test_receive_only_skips_pickup(self, tmp_path):
        session, send_transport, patches = _mocked_session(tmp_path, receive_only=True)

        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            assert await session.start() is True

        send_transport.produce.assert_not_awaited()
        session._signaling.pickup.assert_not_awaited()
        assert session._audio_producer is None
        await session.stop()

    async def test_normal_session_produces_audio(self, tmp_path):
        session, send_transport, patches = _mocked_session(tmp_path, receive_only=False)

        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            assert await session.start() is True

        send_transport.produce.assert_awaited_once()
        await session.stop()

    def test_receive_only_session_disables_hangup(self, tmp_path):
        session = FermaxStreamSession(
            signaling_url="https://signaling-pro-duoxme.fermax.io",
            oauth_token="",
            fcm_token="",
            room_id="r1",
            media_root=str(tmp_path),
            receive_only=True,
        )
        assert session._signaling._send_hangup is False


class TestSignalingHangup:
    """Hangup is only sent for sessions that could have picked up."""

    async def test_disconnect_hangs_up_by_default(self):
        client = FermaxSignalingClient()
        sio = AsyncMock()
        client._sio = sio
        client._connected = True

        await client.disconnect()

        sio.emit.assert_awaited_once_with("hang_up", {})

    async def test_receive_only_disconnect_skips_hangup(self):
        client = FermaxSignalingClient(send_hangup=False)
        sio = AsyncMock()
        client._sio = sio
        client._connected = True

        await client.disconnect()

        sio.emit.assert_not_awaited()


class TestFirstFrameWait:
    """Snapshot waiters get the first frame, or None when none arrives."""

    def _session(self) -> FermaxStreamSession:
        return FermaxStreamSession(
            signaling_url="https://signaler.example",
            oauth_token="tok",
            fcm_token="fcm",
            room_id="room",
        )

    async def test_returns_frame_when_already_received(self):
        session = self._session()
        session._latest_frame = b"jpeg"
        session._first_frame_event.set()

        assert await session.wait_for_first_frame(0.1) == b"jpeg"

    async def test_times_out_without_frame(self):
        session = self._session()

        assert await session.wait_for_first_frame(0.01) is None

    async def test_grabber_end_wakes_waiters(self):
        from aiortc.mediastreams import MediaStreamError

        session = self._session()
        track = MagicMock()
        track.kind = "video"
        track.recv = AsyncMock(side_effect=MediaStreamError())
        session._consumer = MagicMock(track=track)
        session._active = True

        await session._grab_frames()

        assert session._first_frame_event.is_set()
        assert await session.wait_for_first_frame(0.01) is None


class TestRecordingToggle:
    """The record flag gates recording setup."""

    def _session(self, tmp_path, record: bool) -> FermaxStreamSession:
        return FermaxStreamSession(
            signaling_url="https://signaler.example",
            oauth_token="tok",
            fcm_token="fcm",
            room_id="room",
            media_root=str(tmp_path),
            record=record,
        )

    def test_record_false_skips_recording(self, tmp_path):
        session = self._session(tmp_path, record=False)

        session._init_recording()

        assert session._recording_path is None
        assert not hasattr(session, "_recording_frames")

    def test_record_true_initializes_recording(self, tmp_path):
        session = self._session(tmp_path, record=True)

        session._init_recording()

        assert session._recording_path is not None
        assert session._recording_frames == []


class TestHangupDecoupling:
    """send_hangup is independent of receive_only when set explicitly."""

    def _session(self, **kwargs) -> FermaxStreamSession:
        return FermaxStreamSession(
            signaling_url="https://signaler.example",
            oauth_token="tok",
            fcm_token="fcm",
            room_id="room",
            **kwargs,
        )

    def test_receive_only_defaults_to_no_hangup(self):
        session = self._session(receive_only=True)
        assert session._signaling._send_hangup is False

    def test_preview_keeps_hangup_when_overridden(self):
        session = self._session(receive_only=True, send_hangup=True)
        assert session._signaling._send_hangup is True

    def test_attending_session_hangs_up(self):
        session = self._session(receive_only=False)
        assert session._signaling._send_hangup is True
