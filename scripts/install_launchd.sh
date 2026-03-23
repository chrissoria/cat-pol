#!/bin/bash
# Install the cat-pol dataset updater as a weekly launchd job.
#
# Usage:
#   bash scripts/install_launchd.sh          # Install and load
#   bash scripts/install_launchd.sh --remove # Unload and remove
#
# Runs every Sunday at 9:00 AM local time.
# Updates all datasets (SD, SF, Salinas, Oakland, Long Beach, Fresno, Berkeley, Federal).

set -euo pipefail

LABEL="com.catpol.dataset-updater"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHON="$(which python3)"

if [ "${1:-}" = "--remove" ]; then
    echo "Unloading ${LABEL}..."
    launchctl bootout "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo "Removed."
    exit 0
fi

mkdir -p "$LOG_DIR"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${SCRIPT_DIR}/update_datasets.py</string>
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

echo "Installed plist at: $PLIST_PATH"
echo "  Python:  $PYTHON"
echo "  Script:  $SCRIPT_DIR/update_datasets.py"
echo "  Log:     $LOG_DIR/launchd.log"
echo "  Schedule: Sundays at 9:00 AM"
echo "  Sources: SD, SF, Salinas, Oakland, Long Beach, Fresno, Berkeley, Federal"

# Load it
launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
echo "Loaded. Will run every Sunday at 9:00 AM."
echo ""
echo "To test immediately:  python scripts/update_datasets.py --dry-run"
echo "To remove:            bash scripts/install_launchd.sh --remove"
