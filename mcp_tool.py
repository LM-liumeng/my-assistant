import asyncio
import json
import re
import subprocess
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Deepin-Power-Manager",
    instructions=(
        "This MCP service can inspect the current power and battery state of the machine, "
        "switch supported power modes, and toggle CPU Boost or power-saving mode. "
        "For safety, it does not expose high-risk operations such as reboot, shutdown, or suspend."
    )
)

# Modes that were verified to be settable on the current machine.
# The advertised `power` mode may exist on some systems, but this environment failed to set it,
# so the tool only exposes balance/performance for now.
SUPPORTED_SETTABLE_MODES = {"balance", "performance"}

# D-Bus constants
UPOWER_SERVICE = "org.freedesktop.UPower"
UPOWER_PATH = "/org/freedesktop/UPower"
UPOWER_IFACE = "org.freedesktop.UPower"

DEEPIN_POWER_SERVICE = "com.deepin.system.Power"
DEEPIN_POWER_PATH = "/com/deepin/system/Power"
DEEPIN_POWER_IFACE = "com.deepin.system.Power"


def _run_busctl(cmd: List[str], timeout: int = 8) -> Dict[str, Any]:
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return {
            "success": cp.returncode == 0,
            "returncode": cp.returncode,
            "stdout": (cp.stdout or "").strip(),
            "stderr": (cp.stderr or "").strip(),
            "command": " ".join(cmd),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": 124,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "command": " ".join(cmd),
        }
    except Exception as ex:
        return {
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": str(ex),
            "command": " ".join(cmd),
        }


def _parse_busctl_value(output: str) -> Any:
    text = (output or "").strip()

    m = re.match(r'^s\s+"(.*)"$', text)
    if m:
        return m.group(1)

    m = re.match(r'^b\s+(true|false)$', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"

    m = re.match(r'^d\s+([-+]?\d+(?:\.\d+)?)$', text)
    if m:
        return float(m.group(1))

    m = re.match(r'^u\s+(\d+)$', text)
    if m:
        return int(m.group(1))

    m = re.match(r'^x\s+([-+]?\d+)$', text)
    if m:
        return int(m.group(1))

    m = re.match(r'^o\s+"(.*)"$', text)
    if m:
        return m.group(1)

    return text


def _get_property(service: str, path: str, interface: str, prop: str) -> Dict[str, Any]:
    result = _run_busctl([
        "busctl", "get-property",
        service, path, interface, prop
    ])
    if result["success"]:
        result["value"] = _parse_busctl_value(result["stdout"])
    else:
        result["value"] = None
    return result


def _call_method(service: str, path: str, interface: str, method: str,
                 signature_and_args: Optional[List[str]] = None) -> Dict[str, Any]:
    cmd = ["busctl", "call", service, path, interface, method]
    if signature_and_args:
        cmd.extend(signature_and_args)
    return _run_busctl(cmd)


def _set_property(service: str, path: str, interface: str, prop: str,
                  signature: str, value: str) -> Dict[str, Any]:
    return _run_busctl([
        "busctl", "set-property",
        service, path, interface, prop, signature, value
    ])


def _format_result(title: str, payload: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "title": title,
            **payload
        },
        ensure_ascii=False,
        indent=2
    )


@mcp.tool()
async def get_upower_summary() -> str:
    """
    Inspect the overall UPower state.

    Returns:
    - UPower daemon version
    - Whether the lid is closed
    - Whether the machine is currently on battery
    - The configured critical-battery action
    - The current display power device
    - The raw list of enumerated power devices
    """
    return await asyncio.to_thread(_get_upower_summary_sync)


def _get_upower_summary_sync() -> str:
    daemon_version = _get_property(UPOWER_SERVICE, UPOWER_PATH, UPOWER_IFACE, "DaemonVersion")
    lid_is_closed = _get_property(UPOWER_SERVICE, UPOWER_PATH, UPOWER_IFACE, "LidIsClosed")
    on_battery = _get_property(UPOWER_SERVICE, UPOWER_PATH, UPOWER_IFACE, "OnBattery")
    critical_action = _call_method(UPOWER_SERVICE, UPOWER_PATH, UPOWER_IFACE, "GetCriticalAction")
    display_device = _call_method(UPOWER_SERVICE, UPOWER_PATH, UPOWER_IFACE, "GetDisplayDevice")
    devices = _call_method(UPOWER_SERVICE, UPOWER_PATH, UPOWER_IFACE, "EnumerateDevices")

    payload = {
        "success": all([
            daemon_version["success"],
            lid_is_closed["success"],
            on_battery["success"],
            critical_action["success"],
            display_device["success"],
            devices["success"],
        ]),
        "data": {
            "DaemonVersion": daemon_version.get("value"),
            "LidIsClosed": lid_is_closed.get("value"),
            "OnBattery": on_battery.get("value"),
            "CriticalAction": _parse_busctl_value(critical_action["stdout"]) if critical_action["success"] else None,
            "DisplayDevice": _parse_busctl_value(display_device["stdout"]) if display_device["success"] else None,
            "EnumerateDevicesRaw": devices["stdout"],
        },
        "raw": {
            "DaemonVersion": daemon_version,
            "LidIsClosed": lid_is_closed,
            "OnBattery": on_battery,
            "GetCriticalAction": critical_action,
            "GetDisplayDevice": display_device,
            "EnumerateDevices": devices,
        }
    }
    return _format_result("UPower summary", payload)


@mcp.tool()
async def get_power_status() -> str:
    """
    Inspect the current power status.

    Returns:
    - Current power mode
    - CPU Boost state
    - Whether the machine is on battery
    - Battery percentage
    - Power-saving mode state
    - Whether the machine has a battery
    - Whether the machine has a lid switch
    - Supported balance/performance/power-save flags
    """
    return await asyncio.to_thread(_get_power_status_sync)


def _get_power_status_sync() -> str:
    props = [
        "Mode",
        "CpuBoost",
        "OnBattery",
        "BatteryPercentage",
        "PowerSavingModeEnabled",
        "HasBattery",
        "HasLidSwitch",
        "IsBalanceSupported",
        "IsHighPerformanceSupported",
        "IsPowerSaveSupported",
    ]

    raw = {}
    data = {}
    ok = True

    for prop in props:
        res = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, prop)
        raw[prop] = res
        data[prop] = res.get("value")
        ok = ok and res["success"]

    payload = {
        "success": ok,
        "data": data,
        "raw": raw,
    }
    return _format_result("Current power status", payload)


@mcp.tool()
async def get_supported_modes() -> str:
    """
    Inspect which power modes are advertised by the machine and which ones this MCP tool allows.

    Note:
    - This tool only exposes `balance` and `performance` by default.
    - The machine may advertise `power`, but setting it failed in this environment, so it is not exposed.
    """
    return await asyncio.to_thread(_get_supported_modes_sync)


def _get_supported_modes_sync() -> str:
    balance = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "IsBalanceSupported")
    perf = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "IsHighPerformanceSupported")
    power = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "IsPowerSaveSupported")

    advertised = []
    if balance.get("value") is True:
        advertised.append("balance")
    if perf.get("value") is True:
        advertised.append("performance")
    if power.get("value") is True:
        advertised.append("power")

    payload = {
        "success": balance["success"] and perf["success"] and power["success"],
        "data": {
            "advertised_supported_modes": advertised,
            "mcp_allowed_modes": sorted(SUPPORTED_SETTABLE_MODES),
            "note": "The advertised 'power' mode is not enabled here because setting it failed on this machine.",
        },
        "raw": {
            "IsBalanceSupported": balance,
            "IsHighPerformanceSupported": perf,
            "IsPowerSaveSupported": power,
        }
    }
    return _format_result("Supported power modes", payload)


@mcp.tool()
async def set_power_mode(mode: str) -> str:
    """
    Switch the power mode.

    Parameter:
    - mode: currently only `balance` and `performance` are supported.
    """
    return await asyncio.to_thread(_set_power_mode_sync, mode)


def _set_power_mode_sync(mode: str) -> str:
    mode = (mode or "").strip().lower()

    if mode not in SUPPORTED_SETTABLE_MODES:
        return _format_result(
            "Set power mode",
            {
                "success": False,
                "error": f"Unsupported mode: {mode}",
                "allowed_modes": sorted(SUPPORTED_SETTABLE_MODES),
                "note": "The 'power' mode is not exposed because setting it failed in this environment.",
            }
        )

    before = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "Mode")
    action = _call_method(
        DEEPIN_POWER_SERVICE,
        DEEPIN_POWER_PATH,
        DEEPIN_POWER_IFACE,
        "SetMode",
        ["s", mode]
    )
    after = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "Mode")

    payload = {
        "success": action["success"],
        "target_mode": mode,
        "before_mode": before.get("value"),
        "after_mode": after.get("value"),
        "raw": {
            "before": before,
            "action": action,
            "after": after,
        }
    }
    return _format_result("Set power mode", payload)


@mcp.tool()
async def set_cpu_boost(enabled: bool) -> str:
    """
    Toggle CPU Boost.

    Parameter:
    - enabled: `True` to enable, `False` to disable.
    """
    return await asyncio.to_thread(_set_cpu_boost_sync, enabled)


def _set_cpu_boost_sync(enabled: bool) -> str:
    before = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "CpuBoost")
    action = _call_method(
        DEEPIN_POWER_SERVICE,
        DEEPIN_POWER_PATH,
        DEEPIN_POWER_IFACE,
        "SetCpuBoost",
        ["b", "true" if enabled else "false"]
    )
    after = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "CpuBoost")

    payload = {
        "success": action["success"],
        "target_enabled": enabled,
        "before": before.get("value"),
        "after": after.get("value"),
        "raw": {
            "before": before,
            "action": action,
            "after": after,
        }
    }
    return _format_result("Set CPU Boost", payload)


@mcp.tool()
async def set_power_saving_mode(enabled: bool) -> str:
    """
    Toggle power-saving mode.

    Parameter:
    - enabled: `True` to enable, `False` to disable.
    """
    return await asyncio.to_thread(_set_power_saving_mode_sync, enabled)


def _set_power_saving_mode_sync(enabled: bool) -> str:
    before = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "PowerSavingModeEnabled")
    action = _set_property(
        DEEPIN_POWER_SERVICE,
        DEEPIN_POWER_PATH,
        DEEPIN_POWER_IFACE,
        "PowerSavingModeEnabled",
        "b",
        "true" if enabled else "false"
    )
    after = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "PowerSavingModeEnabled")

    payload = {
        "success": action["success"],
        "target_enabled": enabled,
        "before": before.get("value"),
        "after": after.get("value"),
        "raw": {
            "before": before,
            "action": action,
            "after": after,
        }
    }
    return _format_result("Set power-saving mode", payload)


@mcp.tool()
async def get_battery_brief() -> str:
    """
    Get a concise summary of the current battery and power state.

    The response is intentionally compact so an upper-layer assistant can consume it directly.
    """
    return await asyncio.to_thread(_get_battery_brief_sync)


def _get_battery_brief_sync() -> str:
    on_battery = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "OnBattery")
    battery_percentage = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "BatteryPercentage")
    mode = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "Mode")
    cpu_boost = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "CpuBoost")
    power_saving = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "PowerSavingModeEnabled")
    has_battery = _get_property(DEEPIN_POWER_SERVICE, DEEPIN_POWER_PATH, DEEPIN_POWER_IFACE, "HasBattery")

    payload = {
        "success": all([
            on_battery["success"],
            battery_percentage["success"],
            mode["success"],
            cpu_boost["success"],
            power_saving["success"],
            has_battery["success"],
        ]),
        "data": {
            "has_battery": has_battery.get("value"),
            "on_battery": on_battery.get("value"),
            "battery_percentage": battery_percentage.get("value"),
            "mode": mode.get("value"),
            "cpu_boost": cpu_boost.get("value"),
            "power_saving_mode": power_saving.get("value"),
        }
    }
    return _format_result("Battery brief", payload)


if __name__ == "__main__":
    mcp.run(transport="stdio")
