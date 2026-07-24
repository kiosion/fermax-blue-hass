"""Tests for the Fermax Blue coordinator."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.fermax_blue.api import (
    AccessDoor,
    CallLogEntry,
    DeviceInfo,
    FermaxBlueApi,
    Pairing,
)
from custom_components.fermax_blue.const import CALL_MODE_NOTIFY, CALL_MODE_RECORD
from custom_components.fermax_blue.coordinator import (
    PHOTO_RETRY_DELAYS,
    FermaxBlueCoordinator,
    _is_trusted_signaling_url,
)


@pytest.fixture
def mock_hass():
    """Return a mock HomeAssistant instance."""
    hass = MagicMock()
    hass.async_create_task = MagicMock()
    return hass


@pytest.fixture
def mock_api():
    """Return a mock API."""
    api = AsyncMock(spec=FermaxBlueApi)
    api.get_device_info = AsyncMock(
        return_value=DeviceInfo(
            device_id="dev1",
            connection_state="Connected",
            status="ACTIVATED",
            family="MONITOR",
            device_type="VEO-XL",
            subtype="WIFI",
            unit_number=42,
            photocaller=True,
            streaming_mode="video_call",
            is_monitor=True,
            wireless_signal=4,
        )
    )
    api.get_dnd_status = AsyncMock(return_value=False)
    api.set_dnd = AsyncMock()
    api.press_f1 = AsyncMock()
    api.call_guard = AsyncMock()
    api.set_photo_caller = AsyncMock()
    api.get_opening_history = AsyncMock(return_value=[])
    api.ack_notification = AsyncMock()
    return api


@pytest.fixture
def pairing():
    """Return a test pairing."""
    return Pairing(
        device_id="dev1",
        tag="Home",
        installation_id="inst_1",
        access_doors={
            "GENERAL": AccessDoor(
                name="GENERAL",
                title="Portal",
                access_id={"block": 100, "subblock": -1, "number": 0},
                visible=True,
            ),
        },
    )


@pytest.fixture
def coordinator(mock_hass, mock_api, pairing):
    """Create a coordinator with patched HA internals."""
    with patch(
        "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.__init__",
        return_value=None,
    ):
        coord = FermaxBlueCoordinator.__new__(FermaxBlueCoordinator)
        coord.api = mock_api
        coord.pairing = pairing
        coord.hass = mock_hass
        coord.device_info = None
        coord.notification_listener = None
        coord._last_photo = None
        coord._last_photo_id = None
        coord._doorbell_ringing = False
        coord._camera_active = False
        coord._last_divert_response = None
        coord._photo_pending_since = None
        coord._photo_retry_attempt = 0
        coord._photo_retry_unsub = None
        coord._storage_path = None
        coord._call_mode = CALL_MODE_NOTIFY
        coord._auto_response_file = ""
        coord._ring_preview = False
        coord._stream_session = None
        coord._preview_pending = False
        coord._doorbell_reset_unsub = None
        coord._camera_timeout_unsub = None
        coord._dnd_enabled = None
        coord._last_opening = None
        coord._notification_start_time = None
        coord._processed_notifications = []
        coord.update_interval = None
    return coord


class TestStreamingDepsGuard:
    """Optional live-video deps: never wake the intercom when they are missing."""

    @pytest.mark.asyncio
    async def test_preview_skipped_without_deps(self, coordinator, mock_api):
        coordinator.notification_listener = MagicMock()
        coordinator.notification_listener.fcm_token = "tok"

        with patch(
            "custom_components.fermax_blue.coordinator.streaming_deps_available",
            return_value=False,
        ):
            result = await coordinator.start_camera_preview()

        assert result is None
        mock_api.auto_on.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_preview_requests_auto_on_with_deps(self, coordinator, mock_api):
        coordinator.notification_listener = MagicMock()
        coordinator.notification_listener.fcm_token = "tok"
        mock_api.auto_on = AsyncMock(return_value=None)

        with patch(
            "custom_components.fermax_blue.coordinator.streaming_deps_available",
            return_value=True,
        ):
            await coordinator.start_camera_preview()

        mock_api.auto_on.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_stream_skipped_without_deps(self, coordinator):
        coordinator._stream_session = None
        coordinator.stop_stream = AsyncMock()

        with patch(
            "custom_components.fermax_blue.coordinator.streaming_deps_available",
            return_value=False,
        ):
            await coordinator._start_stream("room1", "https://signaling-pro-duoxme.fermax.io")

        coordinator.stop_stream.assert_not_awaited()
        assert coordinator.stream_session is None


class TestCoordinatorDnd:
    """Test DND coordination."""

    @pytest.mark.asyncio
    async def test_set_dnd_calls_api(self, coordinator, mock_api):
        coordinator.notification_listener = MagicMock()
        coordinator.notification_listener.fcm_token = "tok"

        await coordinator.set_dnd(True)
        mock_api.set_dnd.assert_called_once_with("dev1", "tok", enabled=True)
        assert coordinator.dnd_enabled is True

    @pytest.mark.asyncio
    async def test_set_dnd_no_listener(self, coordinator, mock_api):
        coordinator.notification_listener = None

        await coordinator.set_dnd(True)
        mock_api.set_dnd.assert_not_called()


class TestCoordinatorF1:
    """Test F1 coordination."""

    @pytest.mark.asyncio
    async def test_press_f1_calls_api(self, coordinator, mock_api):
        await coordinator.press_f1()
        mock_api.press_f1.assert_called_once_with("dev1")


class TestCoordinatorCallGuard:
    """Test call guard coordination."""

    @pytest.mark.asyncio
    async def test_call_guard_calls_api(self, coordinator, mock_api):
        await coordinator.call_guard()
        mock_api.call_guard.assert_called_once_with("dev1")


class TestCoordinatorFcmWatchdog:
    """Test the FCM listener watchdog hook."""

    @pytest.mark.asyncio
    async def test_no_listener_is_noop(self, coordinator):
        coordinator.notification_listener = None
        await coordinator.ensure_notifications_running()

    @pytest.mark.asyncio
    async def test_delegates_to_listener(self, coordinator):
        listener = MagicMock()
        listener.ensure_running = AsyncMock(return_value=True)
        coordinator.notification_listener = listener
        coordinator._notification_start_time = 12345.0

        await coordinator.ensure_notifications_running()

        listener.ensure_running.assert_awaited_once()
        assert coordinator._notification_start_time == 12345.0


class TestCoordinatorPhotoCaller:
    """Test photo caller coordination."""

    @pytest.mark.asyncio
    async def test_set_photo_caller_calls_api(self, coordinator, mock_api):
        coordinator.device_info = DeviceInfo(
            device_id="dev1",
            connection_state="Connected",
            status="ACTIVATED",
            family="MONITOR",
            device_type="VEO-XL",
            subtype="WIFI",
            unit_number=42,
            photocaller=False,
            streaming_mode="video_call",
            is_monitor=True,
            wireless_signal=4,
        )

        await coordinator.set_photo_caller(True)
        mock_api.set_photo_caller.assert_called_once_with("dev1", enabled=True)
        assert coordinator.device_info.photocaller is True


class TestRingPhotoRecency:
    """Ring photos are accepted by call recency and retried until registered."""

    @pytest.mark.asyncio
    async def test_stale_entry_rejected_then_fresh_accepted(self, coordinator, mock_api):
        ring_time = datetime.now(UTC)
        stale = CallLogEntry(
            call_id="c1",
            device_id="dev1",
            call_date=ring_time - timedelta(minutes=7),
            photo_id="old",
            answered=False,
        )
        fresh = CallLogEntry(
            call_id="c2",
            device_id="dev1",
            call_date=ring_time + timedelta(seconds=2),
            photo_id="new",
            answered=False,
        )
        mock_api.get_call_log = AsyncMock(side_effect=[[stale], [stale, fresh]])
        mock_api.get_call_photo = AsyncMock(return_value=b"jpeg")
        coordinator.hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())
        coordinator.notification_listener = MagicMock()
        coordinator.notification_listener.fcm_token = "tok"
        coordinator._photo_pending_since = ring_time

        unsub = MagicMock()
        with patch(
            "custom_components.fermax_blue.coordinator.async_call_later",
            return_value=unsub,
        ) as mock_later:
            await coordinator._async_update_data()

            # First fetch sees only the previous call. No photo, retry scheduled
            mock_api.get_call_photo.assert_not_awaited()
            assert coordinator._photo_pending_since == ring_time
            mock_later.assert_called_once()
            assert mock_later.call_args.args[1] == PHOTO_RETRY_DELAYS[0]

            await coordinator._async_update_data()

        # Second fetch sees the ring's entry. Photo accepted, retry canceled
        assert coordinator._last_photo == b"jpeg"
        assert coordinator._last_photo_id == "new"
        assert coordinator._photo_pending_since is None
        unsub.assert_called_once()

    @pytest.mark.asyncio
    async def test_photo_id_persisted_across_restart(self, coordinator, tmp_path):
        coordinator._storage_path = tmp_path
        coordinator._last_photo = b"img"
        coordinator._last_photo_id = "p123"
        await coordinator._save_last_photo()

        coordinator._last_photo = None
        coordinator._last_photo_id = None
        await coordinator._load_last_photo()

        assert coordinator._last_photo == b"img"
        assert coordinator._last_photo_id == "p123"

    def test_retry_ladder_gives_up(self, coordinator):
        coordinator._photo_pending_since = datetime.now(UTC)
        coordinator._photo_retry_attempt = len(PHOTO_RETRY_DELAYS)

        with patch("custom_components.fermax_blue.coordinator.async_call_later") as mock_later:
            coordinator._schedule_photo_retry()

        assert coordinator._photo_pending_since is None
        assert coordinator._photo_retry_attempt == 0
        mock_later.assert_not_called()

    def test_no_retry_without_photocaller(self, coordinator):
        coordinator.device_info = DeviceInfo(
            device_id="dev1",
            connection_state="Connected",
            status="ACTIVATED",
            family="MONITOR",
            device_type="VEO-XL",
            subtype="WIFI",
            unit_number=42,
            photocaller=False,
            streaming_mode="video_call",
            is_monitor=True,
            wireless_signal=4,
        )
        coordinator._photo_pending_since = datetime.now(UTC)

        with patch("custom_components.fermax_blue.coordinator.async_call_later") as mock_later:
            coordinator._schedule_photo_retry()

        assert coordinator._photo_pending_since is None
        mock_later.assert_not_called()


class TestRingPreview:
    """The ring preview option starts a receive-only stream without answering."""

    def _ring(self, coordinator, persistent_id="n1"):
        notification = {
            "data": {
                "FermaxNotificationType": "Call",
                "RoomId": "room1",
                "SocketUrl": "https://signaling-pro-duoxme.fermax.io",
                "FermaxToken": "ftok",
            }
        }
        with (
            patch("custom_components.fermax_blue.coordinator.async_dispatcher_send"),
            patch(
                "custom_components.fermax_blue.coordinator.async_call_later",
                return_value=MagicMock(),
            ),
        ):
            coordinator._handle_notification(notification, persistent_id)

    def test_ring_starts_receive_only_stream(self, coordinator):
        coordinator.hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())
        coordinator._ring_preview = True
        coordinator._start_stream = MagicMock()

        self._ring(coordinator)

        coordinator._start_stream.assert_called_once_with(
            "room1", "https://signaling-pro-duoxme.fermax.io", "ftok", receive_only=True
        )

    def test_no_stream_in_notify_mode_by_default(self, coordinator):
        coordinator.hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())
        coordinator._start_stream = MagicMock()

        self._ring(coordinator)

        coordinator._start_stream.assert_not_called()

    def test_attending_call_mode_still_picks_up(self, coordinator):
        coordinator.hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())
        coordinator._ring_preview = True
        coordinator._call_mode = CALL_MODE_RECORD
        coordinator._start_stream = MagicMock()

        self._ring(coordinator)

        assert coordinator._start_stream.call_args.kwargs["receive_only"] is False


class TestEnsureCameraPreview:
    """Viewer-triggered preview starts are single-flight and idempotent."""

    @pytest.mark.asyncio
    async def test_starts_preview_when_idle(self, coordinator):
        coordinator.start_camera_preview = AsyncMock()

        await coordinator.ensure_camera_preview()

        coordinator.start_camera_preview.assert_awaited_once()
        assert coordinator._preview_pending is False

    @pytest.mark.asyncio
    async def test_skips_when_stream_session_exists(self, coordinator):
        coordinator.start_camera_preview = AsyncMock()
        coordinator._stream_session = MagicMock()

        await coordinator.ensure_camera_preview()

        coordinator.start_camera_preview.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_while_camera_active(self, coordinator):
        coordinator.start_camera_preview = AsyncMock()
        coordinator._camera_active = True

        await coordinator.ensure_camera_preview()

        coordinator.start_camera_preview.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_calls_start_once(self, coordinator):
        release = asyncio.Event()

        async def _slow_start():
            await release.wait()

        coordinator.start_camera_preview = AsyncMock(side_effect=_slow_start)

        first = asyncio.create_task(coordinator.ensure_camera_preview())
        await asyncio.sleep(0)
        second = asyncio.create_task(coordinator.ensure_camera_preview())
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(first, second)

        coordinator.start_camera_preview.assert_awaited_once()


class TestCoordinatorScanInterval:
    """Test configurable scan interval."""

    def test_default_interval(self, mock_hass, mock_api, pairing):
        with patch(
            "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.__init__"
        ) as mock_init:
            FermaxBlueCoordinator(mock_hass, mock_api, pairing)
            call_kwargs = mock_init.call_args
            assert call_kwargs.kwargs["update_interval"].total_seconds() == 300

    def test_custom_interval(self, mock_hass, mock_api, pairing):
        with patch(
            "homeassistant.helpers.update_coordinator.DataUpdateCoordinator.__init__"
        ) as mock_init:
            FermaxBlueCoordinator(mock_hass, mock_api, pairing, scan_interval=10)
            call_kwargs = mock_init.call_args
            assert call_kwargs.kwargs["update_interval"].total_seconds() == 600


class TestSignalingUrlValidation:
    """Test signaling URL domain validation."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://signaling-pro-duoxme.fermax.io",
            "https://signaling.fermax.io/path",
            "wss://signaling-pro-duoxme.fermax.io",
            "https://fermax.io",
        ],
    )
    def test_trusted_urls_accepted(self, url):
        assert _is_trusted_signaling_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.com",
            "https://notfermax.io",
            "https://fermax.io.evil.com",
            "https://evil-fermax.io",
            "",
            "not-a-url",
        ],
    )
    def test_untrusted_urls_rejected(self, url):
        assert _is_trusted_signaling_url(url) is False
