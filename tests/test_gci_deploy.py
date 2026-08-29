from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_systemd_unit_is_independent_and_resource_bounded():
    unit = (ROOT / "deploy" / "harness-gci.service").read_text()
    assert "ExecStart=/home/samkim/harness-gci/venv/bin/harness gci serve" in unit
    assert "EnvironmentFile=/etc/harness-gci.env" in unit
    assert "Restart=on-failure" in unit
    assert "Nice=10" in unit
    assert "IOSchedulingClass=idle" in unit
    assert "MemoryMax=4G" in unit
    assert "CPUQuota=100%" in unit
    assert "ReadWritePaths=/data/harness-gci" in unit
    assert "bge-m3-embed.service" not in unit
    assert "8800" not in unit


def test_deploy_script_only_manages_gci_unit():
    script = (ROOT / "scripts" / "deploy_gci.sh").read_text()
    assert "systemctl enable harness-gci.service" in script
    assert "systemctl restart harness-gci.service" in script
    assert "systemctl restart bge" not in script
    assert "systemctl stop bge" not in script
    assert "systemctl start bge" not in script


def test_launchd_refresh_job_is_short_lived_and_source_side():
    template = (
        ROOT / "scripts" / "com.samkim.harness-gci-refresh.plist.template"
    ).read_text()
    assert "com.samkim.harness-gci-refresh" in template
    assert "<string>gci</string>" in template
    assert "<string>auto-run</string>" in template
    assert "<key>StartInterval</key>" in template
    assert "<integer>300</integer>" in template
    assert "<key>KeepAlive</key>" not in template
    assert "spark" not in template.lower()
    assert "git pull" not in template


def test_launchd_installer_is_idempotent_and_user_scoped():
    script = (ROOT / "scripts" / "install_gci_refresh.sh").read_text()
    assert "set -euo pipefail" in script
    assert "Library/LaunchAgents" in script
    assert 'bootout "$DOMAIN/$LABEL"' in script
    assert 'bootstrap "$DOMAIN" "$PLIST"' in script
    assert "sudo" not in script
    assert "git pull" not in script
