#!/bin/bash
# Install the cat-pol dataset updaters as launchd jobs.
#
# Two schedules:
#   1. Weekly (Sundays 9 AM) — all sources except Truth Social
#   2. Daily  (every day 9 AM) — Trump Truth Social only
#
# Usage:
#   bash scripts/install_launchd.sh          # Install and load both
#   bash scripts/install_launchd.sh --remove # Unload and remove both

set -euo pipefail

LABEL_WEEKLY="com.catpol.dataset-updater"
LABEL_DAILY_TS="com.catpol.truthsocial-updater"
LABEL_CA_BILLS="com.catpol.california-bills-updater"
PLIST_WEEKLY="$HOME/Library/LaunchAgents/${LABEL_WEEKLY}.plist"
PLIST_DAILY_TS="$HOME/Library/LaunchAgents/${LABEL_DAILY_TS}.plist"
PLIST_CA_BILLS="$HOME/Library/LaunchAgents/${LABEL_CA_BILLS}.plist"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHON="$(which python3)"

if [ "${1:-}" = "--remove" ]; then
    echo "Unloading ${LABEL_WEEKLY}..."
    launchctl bootout "gui/$(id -u)" "$PLIST_WEEKLY" 2>/dev/null || true
    rm -f "$PLIST_WEEKLY"
    echo "Unloading ${LABEL_DAILY_TS}..."
    launchctl bootout "gui/$(id -u)" "$PLIST_DAILY_TS" 2>/dev/null || true
    rm -f "$PLIST_DAILY_TS"
    echo "Unloading ${LABEL_CA_BILLS}..."
    launchctl bootout "gui/$(id -u)" "$PLIST_CA_BILLS" 2>/dev/null || true
    rm -f "$PLIST_CA_BILLS"
    echo "Removed all."
    exit 0
fi

mkdir -p "$LOG_DIR"

# ---- Weekly job: all sources except Truth Social ----

# Build the --city list excluding ts
WEEKLY_CITIES="sd sf sal oak lb fre ber bak la clovis nb sdco fed eo speeches"

cat > "$PLIST_WEEKLY" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL_WEEKLY}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${SCRIPT_DIR}/update_datasets.py</string>
        <string>--exclude</string>
        <string>ts</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
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
    <string>${LOG_DIR}/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

echo "Installed weekly plist at: $PLIST_WEEKLY"
echo "  Schedule: Sundays at 9:00 AM (all sources except Truth Social)"

# ---- Daily job: Truth Social only ----

cat > "$PLIST_DAILY_TS" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL_DAILY_TS}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${SCRIPT_DIR}/update_datasets.py</string>
        <string>--city</string>
        <string>ts</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd_truthsocial.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd_truthsocial.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

echo "Installed daily plist at: $PLIST_DAILY_TS"
echo "  Schedule: Every day at 9:00 AM (Truth Social only)"

# ---- Sunday job: California state bills ----

cat > "$PLIST_CA_BILLS" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL_CA_BILLS}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${SCRIPT_DIR}/update_california_bills.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd_ca_bills.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd_ca_bills.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

echo "Installed CA bills plist at: $PLIST_CA_BILLS"
echo "  Schedule: Sundays at 9:30 AM (California state bills)"

# Load all three
launchctl bootstrap "gui/$(id -u)" "$PLIST_WEEKLY"
launchctl bootstrap "gui/$(id -u)" "$PLIST_DAILY_TS"
launchctl bootstrap "gui/$(id -u)" "$PLIST_CA_BILLS"
echo ""
echo "All jobs loaded."
echo "  Python:  $PYTHON"
echo "  Logs:    $LOG_DIR/launchd.log (weekly), $LOG_DIR/launchd_truthsocial.log (daily), $LOG_DIR/launchd_ca_bills.log (Sunday)"
echo ""
echo "To test immediately:"
echo "  python scripts/update_datasets.py --dry-run --exclude ts   # weekly sources"
echo "  python scripts/update_datasets.py --dry-run --city ts      # Truth Social"
echo "  python scripts/update_california_bills.py --dry-run         # CA bills"
echo "To remove:  bash scripts/install_launchd.sh --remove"
