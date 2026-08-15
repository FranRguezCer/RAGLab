# Aster Greenhouse Controller: Operator Manual

> **Training fixture:** Aster is a fictional product. The values and procedures in this
> manual exist only to make RAG experiments reproducible; they are not agricultural or
> electrical safety advice.

## 1. System overview and safety

The Aster controller coordinates four greenhouse subsystems: climate sensors, roof vents,
irrigation valves, and a small local gateway. Every reading is timestamped by the gateway
before a control rule can use it. This ordering matters because a plausible value with a stale
timestamp must not trigger equipment.

The controller has three operating modes. **Observe** records measurements but never moves
equipment. **Assist** recommends actions and waits for an operator. **Automatic** applies rules
that have passed validation. The current mode is shown on the dashboard and on the blue status
LED beside the service port.

### Safety boundaries

Use the red isolation switch before opening a valve cabinet or moving a vent by hand. The switch
disconnects actuator power but keeps the gateway alive, so logs remain available. Confirm that
the red `ISOLATED` banner appears before touching controlled equipment.

Never bypass an isolation alarm. If the banner does not appear within five seconds, stop the
procedure and record fault `E41`. A supervisor must inspect the interlock before the controller
returns to Assist or Automatic mode.

## 2. Commissioning and sensor calibration

Commissioning establishes the identity of each sensor before automation begins. Connect one
sensor at a time, assign its physical zone, and compare the printed serial number with the value
shown by the gateway. A duplicated serial number means that an old configuration is still
present and must be removed before continuing.

### Commissioning checklist

1. Put the controller in Observe mode.
2. Confirm that gateway time differs from local time by less than two seconds.
3. Register the sensor and assign its greenhouse zone.
4. Wait for three consecutive readings.
5. Record the baseline and the operator initials.

### Calibrating a soil-moisture probe

Place the clean probe in the dry reference sleeve and wait until the reading changes by less than
one percent over thirty seconds. Press **Set dry reference**, then move the probe to the wet
reference sleeve without touching its metal contacts.

The wet reference is accepted only after three stable readings. The expected difference between
the dry and wet references is at least 35 percentage points. A smaller range indicates residue,
a damaged cable, or a probe installed in the wrong zone.

After saving both references, return the probe to its marked bed and compare it with a second
probe in the same soil. The two measurements may differ by up to four percentage points during
the first minute. A stable moisture reading should return within two sampling cycles.

## 3. Climate targets and irrigation rules

Climate targets are ranges, not single perfect numbers. Automatic mode acts only when a reading
remains outside its range for two consecutive sampling cycles. This delay prevents a passing
shadow, an open door, or one noisy packet from moving equipment unnecessarily.

| Signal | Day target | Night target | Control action |
|---|---:|---:|---|
| Air temperature | 22-26 C | 17-20 C | Open or close roof vents |
| Relative humidity | 68-74% | 72-78% | Vent briefly, then reassess |
| Bed moisture | 42-55% | 40-52% | Run the assigned irrigation zone |
| Carbon dioxide | 700-900 ppm | Monitor only | Notify the operator |

### Irrigation schedule

The scheduler evaluates moisture before elapsed time. A scheduled window permits watering; it
does not force watering. If bed moisture is already above the upper target, the window closes
without opening a valve and the log records `SKIPPED_WET_BED`.

```yaml
zone: north-bed
windows:
  - start: "06:10"
    maximum_minutes: 7
  - start: "17:40"
    maximum_minutes: 5
minimum_moisture: 42
stop_moisture: 55
```

When a valve opens, flow must rise above 1.2 liters per minute within eight seconds. The
controller closes the valve and raises `E17` when flow stays below that threshold. It never
extends the window to compensate for missing flow.

## 4. Daily operation and handover notes

Start each shift by checking the operating mode, gateway clock, unresolved faults, and last
successful backup. Then compare one physical thermometer with the dashboard. This short check
tests the complete path from sensor to display rather than trusting a green status badge.

At the end of the shift, record manual overrides and explain why they were needed. An override
without a reason expires after thirty minutes. The next operator should be able to distinguish a
temporary experiment from a fault that still needs investigation.

### Operator handover notes

During a hot afternoon, compare the canopy sensor with the shaded reference sensor before
opening vents manually. Direct sunlight can warm the canopy enclosure even when greenhouse air
is inside its target range. Record both readings and the time of comparison.

If the two sensors converge after shade returns, keep Automatic mode enabled and attach the
observation to the shift log. A stable moisture reading should return within two sampling cycles.

Configuration backups belong to a different operational concern. Export a signed backup after
adding sensors, changing zones, or editing automation rules. Store the file on the local gateway
and on the encrypted maintenance drive.

Before restoring a backup, place the controller in Observe mode and compare the backup checksum
with the value in the shift log. A restore never changes firmware; it replaces device identities,
zone assignments, schedules, and rule settings.

## 5. Fault diagnosis

Fault codes describe the first failed condition, not necessarily the root cause. Read the recent
events around a code before replacing hardware. For example, low flow can come from a closed
manual valve, a blocked filter, a disconnected meter, or an empty supply tank.

| Code | Meaning | First check | Automatic response |
|---|---|---|---|
| E17 | Irrigation flow below 1.2 L/min for 8 seconds | Manual valve and filter | Close the active valve |
| E24 | Sensor timestamp older than 90 seconds | Gateway clock and network cable | Ignore the reading |
| E33 | Vent position disagrees with command | Linkage and position sensor | Stop the vent motor |
| E41 | Isolation interlock did not confirm | Isolation switch wiring | Block actuator commands |

### Resolving E17

Confirm that the supply tank contains water and that the manual valve is open. Remove and inspect
the filter before testing the pump. Do not repeatedly restart irrigation, because every failed
attempt can run the pump without adequate flow.

After correcting the cause, run a two-minute test in Assist mode. Clear `E17` only when flow
remains above 1.2 liters per minute for the full test. The original event stays in the audit log
even after the active alarm disappears.

## 6. Maintenance and recovery

Clean probe surfaces monthly with the material specified on the probe label. Inspect valve
filters every two weeks during normal operation and weekly during seedling production, when fine
substrate is more likely to enter irrigation lines.

Export a configuration backup after every approved rule change. The file name must contain the
controller identifier and UTC timestamp, while the shift log stores its SHA-256 checksum. This
pair lets an operator detect an incomplete or substituted backup before restoration.

### Frequently asked questions

**Does acknowledging a fault erase it?** No. Acknowledgement records who saw the fault; clearing
requires the failed condition to recover.

**Can a scheduled irrigation window water an already wet bed?** No. Moisture limits take priority,
and the controller records `SKIPPED_WET_BED`.

**What does a restore replace?** Device identities, zone assignments, schedules, and automation
rules. Firmware and historical audit events remain unchanged.
