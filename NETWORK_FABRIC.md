# DGX Spark 200G network fabric

Evidence date: 2026-08-31

## Scope and current state

The MikroTik `CRS812-8DS-2DQ-2DDQ` is the shared high-speed fabric for the six
non-protected DGX Spark/GX10 systems. `DGX1`/`spark-e10b` remains outside this
fabric and must not be recabled, restarted, or repurposed.

The switch runs RouterOS 7.24.1 and currently receives `192.168.4.50/22` by
DHCP on its management bridge through `ether2`. Treat that address as a lease
until the router has a reservation for the switch. Never store its RouterOS
password in this repository or in command-line history.

All six non-protected nodes are connected and operational:

- `qsfp56-1-1` → ASUS1 (`gx10-fc2e`), NADDOD `Q56-200G-CU1`, 200 Gb/s,
  auto-negotiated `fec91`.
- `qsfp56-2-1` → DGX2 (`spark-49af`), NADDOD `Q56-200G-CU1`, 200 Gb/s,
  auto-negotiated `fec91`.
- `qsfp56-dd-1-1` → DGX3 (`spark-69c8`), Tensor Juice breakout leg 1, forced
  `200G-baseCR4` with `fec91`.
- `qsfp56-dd-1-5` → ASUS3 (`gx10-0309`), Tensor Juice breakout leg 2, forced
  `200G-baseCR4` with `fec91`.
- `qsfp56-dd-2-1` → ASUS2 (`gx10-26b6`), Tensor Juice breakout leg 1, forced
  `200G-baseCR4` with `fec91`.
- `qsfp56-dd-2-5` → ASUS4 (`gx10-33af`), Tensor Juice breakout leg 2, forced
  `200G-baseCR4` with `fec91`.

The switch comments record this physical mapping. All six active masters are
forwarding in the same bridge with Marvell hardware offload.

Both Tensor Juice 400G-to-2x200G breakout modules report
`eeprom-checksum=bad` to RouterOS. A prior investigation incorrectly concluded
that this made the cables unusable. A controlled retest proved that all four
breakout legs carry traffic at 200 Gb/s with RS-FEC when both group masters in
each cage are forced to `200G-baseCR4` and `fec91`. Auto-negotiation left the
second leg at 100 Gb/s without FEC or failed to train it. Treat the checksum as
a module-management compliance defect and retain the ordered replacement
cables as rollback assets; do not treat it as evidence that the active links
are down. The NADDOD straight cables negotiate 200 Gb/s and `fec91`
automatically.

## Existing pair addresses

The switch is transparent at Layer 2, so the existing isolated addresses and
host interface names remain unchanged:

- Qwen pair:
  - ASUS2: `10.10.10.1/24` on `enp1s0f1np1`
  - ASUS2: `10.10.11.1/24` on `enP2p1s0f1np1`
  - ASUS4: `10.10.10.2/24` and `10.10.11.2/24`
- Four-node training pool:
  - DGX2: `10.77.0.1/24` on `enp1s0f1np1`
  - DGX2: `10.77.1.1/24` on `enP2p1s0f1np1`
  - ASUS1: `10.77.0.2/24` and `10.77.1.2/24`
  - DGX3: `10.77.0.3/24` and `10.77.1.3/24`
  - ASUS3: `10.77.0.4/24` and `10.77.1.4/24`

Both logical ConnectX interfaces see the same physical cable module. Their
presence does not imply 400 Gb/s of aggregate physical bandwidth; each node's
current switch attachment is one 200 Gb/s physical link.

All host profiles use MTU 9000 and no default route. Fabric bridge masters use
an L2 MTU of 9216. The bridge CPU/management MTU remains 1500, which does not
limit hardware-offloaded forwarding between the high-speed ports.

## Qualification completed

The six active switch ports report:

- `status=link-ok`
- `rate=200Gbps`
- `fec=fec91`
- active hardware-offloaded bridge forwarding

Both logical interfaces on every connected host report 200,000 Mb/s, full
duplex, link detected, and active RS-FEC. The four training nodes passed
all-to-all 8972-byte do-not-fragment ICMP with zero loss on both training
subnets. The Qwen pair passed the same jumbo test on both Qwen subnets.

Qwen was restored successfully after recabling and its health endpoint
returned HTTP 200. A distributed completion returned exactly `BREAKOUT_OK`.
DGX3 vision and ASUS3 Nemotron were deliberately paused for the four-rank
qualification below, then restored and verified at HTTP 200.

The post-switch two-rank NCCL qualification completed successfully on
DGX2/ASUS1. NCCL selected both RoCE HCAs, all reductions were correct, and the
256 MiB result was 22.9247 Gb/s versus 23.0570 Gb/s on the prior direct-link
run (-0.57%). The 64 MiB result was 22.5745 Gb/s versus 22.2665 Gb/s (+1.38%).
This establishes no meaningful switched-path regression; it does not claim
that this Python/PyTorch all-reduce smoke saturates the 200 Gb/s wire rate.
The sealed result is
`results/post-switch-nccl-20260901.json` (SHA-256
`7cefb9075733acc0e8041c35c16b474fd7905c0e8238fde1037c77cac6fc6afa`).

The expected performance for existing two-node workloads is approximately
unchanged. The switch adds a four-node training topology and aggregate
multi-pair capacity, not extra bandwidth to one two-node job.

## Four-node training qualification

NVIDIA's supported four-Spark topology is one cable per node through a switch;
three nodes can instead use a direct ring. The DGX2, ASUS1, DGX3, and ASUS3
network foundation matches the four-node switch topology: static private
ConnectX addresses, MTU 9000, 200 Gb/s RS-FEC links, hardware-offloaded
switching, and all-to-all jumbo connectivity.

The four-rank NCCL qualification passed with both RoCE HCAs selected on all
four nodes and correct reductions on every iteration. The 256 MiB result was
15.3673 Gb/s algorithm bandwidth and 23.0510 Gb/s bus bandwidth; the 64 MiB
result was 16.1150 Gb/s algorithm bandwidth and 24.1726 Gb/s bus bandwidth.
The sealed result is `results/four-node-switch-nccl-20260901.json` (SHA-256
`7a69703d6774b56c9c0bbbea5b3a71e0eef5a182abdf10561e573d9594009131`).

The first attempt exposed a host configuration difference: DGX2 and ASUS1
have their IPv4-mapped RoCE-v2 GIDs at index 5, while DGX3 and ASUS3 have them
at index 3. A global index 3 therefore mixed link-local IPv6 and IPv4 GIDs and
failed `ibv_modify_qp`. The qualified launcher accepts a per-node
`--gid-index`; use 5 on DGX2/ASUS1 and 3 on DGX3/ASUS3 unless a fresh GID-table
inspection proves otherwise.

An application-level Qwen3.8 native-TP qualification subsequently passed on
the same four ranks. The 95.37 GiB PLE table was column-sharded four ways,
all ranks completed a 32-token forward, and a sequence-8 LoRA smoke completed
one optimizer step with identical adapter state on every rank. Load time was
195–210 seconds, optimizer-step time was 5.80–5.82 seconds, and peak CUDA
allocation was 84.894 GiB per rank. This proves correctness, not fabric
saturation or long-sequence capacity; the minimum post-step free-memory
reading was 0.908 GiB.

NVIDIA Sync's Cluster Assistant configures network and SSH only; it does not
configure NCCL, DeepSpeed, FSDP, or the training launcher. The physical switch
has six members, which exceeds the assistant's four-node support limit, so
preserve the manually validated profiles rather than running the assistant
blindly.

## Operational checks

On RouterOS, monitor only the group masters:

```text
/interface ethernet monitor qsfp56-1-1 once
/interface ethernet monitor qsfp56-2-1 once
/interface ethernet monitor qsfp56-dd-1-1 once
/interface ethernet monitor qsfp56-dd-1-5 once
/interface ethernet monitor qsfp56-dd-2-1 once
/interface ethernet monitor qsfp56-dd-2-5 once
```

A usable 200G result requires `link-ok`, `200Gbps`, and `fec91`; merely seeing
`running=yes` or an EEPROM checksum warning is insufficient.

On each host:

```shell
ethtool enp1s0f1np1
ethtool enP2p1s0f1np1
ethtool --show-fec enp1s0f1np1
ibdev2netdev
```

If any Tensor Juice breakout leg falls back to 100 Gb/s or loses FEC, confirm
that RouterOS still has auto-negotiation disabled, speed forced to
`200G-baseCR4`, and FEC forced to `fec91` on both group masters in that cage.
Do not recable a live distributed process; stop it cleanly first because
NCCL/SGLang will not survive the link loss.

