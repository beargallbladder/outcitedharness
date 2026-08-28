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
