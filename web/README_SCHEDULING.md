# Scheduling the Update pipeline

There are two ways to run Updates automatically, matched to different needs.

## Option A: built-in scheduler (default, already wired up)

The dashboard's "Automatic scheduling" section lets you enable a recurring
Update directly in the app -- no setup needed beyond checking the box and
setting an interval. It's implemented as a background thread inside
`app.py` (see `job_runner.py`'s `_scheduler_loop`) that checks once a
minute whether the next scheduled run is due, and if so calls the exact
same `start_update_job()` the manual Update button uses.

**Limitation:** this only works while `python3 app.py` is actually running.
If you close the terminal, quit the process, or your Mac goes to sleep,
no scheduled run happens until the app is running again. For a local,
occasional-use tool this is usually fine -- but if you want a schedule
that survives your Mac sleeping or the app not being manually kept open,
use Option B.

## Option B: macOS `launchd` (more robust, not installed automatically)

`launchd` is macOS's own service scheduler. A user agent can wake your Mac
at a set time (via `pmset`, optional) and run a command -- in this case,
hitting the app's `/jobs/update` endpoint. This changes your machine's
startup/scheduling behavior, so it's documented here for you to review and
install yourself rather than done automatically.

**1. Make sure `app.py` is actually running when the schedule fires.** The
simplest approach: also let launchd keep the app itself running (so it survives
reboots), OR run this only when you know you'll have the app open.

**2. Create `~/Library/LaunchAgents/com.exoplanetai.update.plist`:**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.exoplanetai.update</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/curl</string>
        <string>-s</string>
        <string>-X</string>
        <string>POST</string>
        <string>http://127.0.0.1:5050/jobs/update</string>
        <string>-d</string>
        <string>sample_size=300</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/exoplanetai_scheduled_update.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/exoplanetai_scheduled_update.log</string>
</dict>
</plist>
```

This example fires every Sunday at 9:00 AM. Adjust `Weekday`/`Hour`/`Minute`,
or replace `StartCalendarInterval` with `StartInterval` (seconds) for a
fixed-period schedule instead of a specific time.

**3. Load it:**
```bash
launchctl load ~/Library/LaunchAgents/com.exoplanetai.update.plist
```

**4. To wake the Mac specifically for this (optional, needs sudo):**
```bash
sudo pmset repeat wakeorpoweron MTWTFSS 08:55:00
```

**5. To remove it later:**
```bash
launchctl unload ~/Library/LaunchAgents/com.exoplanetai.update.plist
rm ~/Library/LaunchAgents/com.exoplanetai.update.plist
```

If `app.py` isn't running when the `curl` fires, the request will simply
fail (connection refused) and get logged to
`/tmp/exoplanetai_scheduled_update.log` -- nothing destructive happens,
it just won't trigger an Update that time.

---

## Option C: let `launchd` run `app.py` itself (recommended for unattended use)

Options A and B both assume `app.py` is already running. This option makes
launchd own the process, which is the piece the app **cannot** do for itself:
the scheduler lives in a `daemon=True` thread inside `app.py`, so if the Flask
main thread dies the scheduler dies with it and nothing brings it back.
`KeepAlive` is what actually restarts it. `/health` only makes the failure
*visible*; this is what makes it *recover*.

A ready-made plist is in the repo at `web/com.exoplanetai.app.plist`. **Its
paths are absolute** -- launchd does not expand `~` and inherits no shell -- so
edit them if the repo lives somewhere else.

**Install:**
```bash
cp web/com.exoplanetai.app.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.exoplanetai.app.plist
```

`bootstrap` is the modern replacement for `launchctl load`, which still works
but is deprecated.

**Check it:**
```bash
launchctl print gui/$(id -u)/com.exoplanetai.app | grep -E "state|pid|runs"
curl -s http://127.0.0.1:5050/health
```

A launchd-owned process has **PPID 1**. That is how you tell it apart from a
leftover `nohup python3 app.py` started by hand:
```bash
lsof -ti tcp:5050 | xargs ps -o pid,ppid,command -p
```

**Verify auto-restart actually works** (worth doing once -- it is the whole
point of this option):
```bash
kill -9 $(lsof -ti tcp:5050)
sleep 40 && curl -s http://127.0.0.1:5050/health
```
It should come back with a new PID. `ThrottleInterval` is 30s, so allow ~30-40
seconds; that delay is deliberate, to stop a genuinely broken app from becoming
a tight respawn loop.

**Stop / uninstall:**
```bash
launchctl bootout gui/$(id -u)/com.exoplanetai.app
rm ~/Library/LaunchAgents/com.exoplanetai.app.plist
```

**What this does NOT solve.** launchd restarts the app if the *process* dies.
It does not keep the *machine* awake. A Mac that sleeps -- including a
battery-triggered "Low Power Sleep", which `caffeinate -i` does not block --
stops the scheduler for the duration, and that time is lost permanently: the
24h retrain gate is keyed to a persisted timestamp and does not catch up. For
genuinely unattended operation, keep the machine on AC power, or run this on an
always-on host.

**Monitoring it from outside:**
```bash
curl -sf http://127.0.0.1:5050/health || echo "scheduler stalled"
tail -f web/logs/scheduler.log
```
`/health` returns **503** when the last tick is more than 300s old, so plain
`curl -f` is enough for any uptime monitor -- no JSON parsing needed.
