# Agent Sandboxing

## Different techniques for agent sandboxing

| Solution | Isolation Level | Platforms Supported | Offline After Setup? | Languages / Runtimes | Setup Difficulty | Key Pros | Key Cons / Limitations | Recommended For |
|---|---|---|---|---|---|---|---|---|
| WebAssembly + Pyodide / wasmtime / wasmer | Medium–High (memory + capability sandbox) | Browser, Linux, macOS, Windows | Yes | Python (Pyodide), Rust, Go, C/C++, JS/TS | Low–Medium | Extremely fast startup, tiny footprint, no kernel access, works in browser too | No real filesystem/network by default (can allow limited), pure-Python deps only in Pyodide | Quick Python/JS experiments, browser-based agents |
| Docker + strict seccomp / AppArmor / rootless | Medium | Linux (best), macOS/Windows via Docker Desktop | Yes | Any (Python, Node, Go, etc.) | Low–Medium | Familiar, huge ecosystem, easy resource limits | Container escapes possible if misconfigured (not VM-level) | General-purpose, when you control the image somewhat |
| Podman rootless + quadlets | Medium–High | Linux (native), macOS via podman machine | Yes | Any | Medium | Daemonless, rootless by default, better security than Docker | Slightly steeper learning curve than Docker | Linux users who want better defaults than Docker |
| Firecracker microVM (local) | Very High (separate kernel) | Linux (easiest), macOS (via lima/colima + firecracker) | Yes | Any Linux-compatible | Medium–High | Hardware-level isolation (KVM), very small guest OS possible | Needs KVM/VT-x, setup is more involved | Highest security needed on Linux |
| Kata Containers (local runtime) | Very High | Linux | Yes | Any | High | Runs containers inside microVMs (Cloud-Hypervisor / QEMU) | Requires container runtime + hypervisor stack | When you want container UX + VM isolation |
| gVisor (runsc runtime) | High (user-space kernel) | Linux | Yes | Any | Medium–High | Intercepts syscalls in user space → strong against kernel exploits | Slower for I/O heavy workloads, not all syscalls supported | Good middle-ground between containers & VMs |
| macOS native Sandbox + App Sandbox / Seatbelt | Medium–High (system services) | macOS only | Yes | Any (via tools like cargo-safe, bubblewrap-like) | Medium | Built-in to macOS, very cheap | Limited to macOS, profile writing is fiddly | macOS users (e.g. cargo-safe for Rust) |
| Windows Sandbox | High–Very High | Windows 10/11 Pro+ | Yes | Any (inside lightweight Hyper-V VM) | Low | One-click, disposable VM, integrated | Only on Pro/Enterprise editions, no GPU passthrough by default | Windows users needing quick & strong isolation |
| libkrun / microsandbox / BoxLite-style tools | Very High | Linux (KVM), macOS (Hypervisor.framework) | Yes | Any Linux-compatible | High (often experimental) | Near-native speed + VM isolation | Cutting-edge / less mature tooling | Bleeding-edge local AI agent sandboxes |

# Docker + strict seccomp / AppArmor / rootless
## 1. Enable rootless Docker (once)
dockerd-rootless-setuptool.sh install

## 2. Create a strict seccomp profile (save as strict.json)
```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    { "names": ["accept", "accept4", "read", "write", ...], "action": "SCMP_ACT_ALLOW" }
    # (most people start from Docker default and remove risky calls)
  ]
}
```

## 3. Run a sandbox container for untrusted code
docker run --rm \
  --user 1000:1000 \
  --cap-drop=ALL \
  --security-opt seccomp=./strict.json \
  --security-opt apparmor=docker-default \
  --security-opt no-new-privileges:true \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  --memory=2g --cpus=2 \
  --network none \                     # or --network slirp4netns for limited net
  python:3.12-slim \
  python -c "print('LLM-generated code runs here safely')"


# Podman Equivalent (Recommended for Most New Setups)

```ini
# ~/.config/containers/systemd/my-sandbox.container
[Container]
Image=python:3.12-slim
Exec=python -c "print('Safe LLM code here')"
User=1000:1000
ReadOnly=true
Tmpfs=/tmp:size=100m,noexec
SecurityOpt=seccomp:strict.json
SecurityOpt=apparmor:podman-default
NoNewPrivileges=true
Memory=2G
CPUQuota=200%

# Enable and start
systemctl --user daemon-reload
systemctl --user enable --now my-sandbox
```

You can now call this from your LangGraph agent via podman exec or systemd API — super clean and production-grade.




# Comparison: Docker Hardened vs Podman Rootless + Quadlets

| Feature | Docker + strict seccomp/AppArmor/rootless | Podman Rootless + Quadlets (2026) | Winner for Agent Sandbox |
|---|---|---|---|
| Daemon | Required (even in rootless mode) | Completely daemonless | Podman (smaller attack surface) |
| Rootless by default | No – must be manually enabled | Yes – default and seamless | Podman |
| Management style | docker run or docker-compose | Declarative `.container` files + systemd (Quadlets) | Podman (much cleaner on servers) |
| Systemd integration | Manual (generate systemd units) | Native & excellent (auto-restart, logging, timers, rollback) | Podman |
| Security for untrusted code | Excellent when fully hardened | Excellent + slightly better default isolation | Tie (both strong) |
| Networking in rootless | slirp4netns (slow) | Same or pasta (newer, faster) | Slight edge to Podman |
| Setup complexity | Medium (extra rootless setup step) | Very low (just install Podman) | Podman |
| Ecosystem / tools | Huge (VS Code, CI, etc.) | Very good + growing fast | Docker (still) |
| Resource overhead | Slightly higher (daemon) | Lower | Podman |
| Port binding <1024 | Possible with extra capabilities | Needs sysctl or rootful for privileged ports | Docker (easier) |