#!/usr/bin/env bash
# One-time setup of a PERSISTENT gpu-unlock boot entry (2026-11-01). Run:  sudo bash ~/ai/gpu-unlock/setup.sh
#
# Why the old trial boot failed: the hand-made "GPU1 BF16 trial" entry booted a copied 7.3.0-5 kernel whose
# /lib/modules tree was removed by the 7.3.1-1 upgrade (vfat -> /boot mount failed -> emergency mode).
#
# What this does (nothing else is touched; your normal, LTS and fallback entries stay as they are):
#  1. removes that stale hand-made entry (and moves its copied kernel to /var/backups)
#  2. installs /etc/gpu-unlock-entry.conf (TARGETS = both 3090s) and the post hook
#     /etc/boot/hooks/post.d/95-gpu-unlock-entry, which limine-mkinitcpio runs after EVERY kernel/initramfs
#     update; it (re)creates the entry "linux-cachyos-gpu-unlock" with limine-entry-tool, so the entry always
#     boots the current kernel with correct BLAKE2 hashes
#  3. runs `limine-mkinitcpio linux-cachyos` once (the same path a kernel update takes) and shows the result
# Undo: sudo bash ~/ai/gpu-unlock/setup.sh --uninstall
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
HOOK=/etc/boot/hooks/post.d/95-gpu-unlock-entry
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p /var/backups
cp -a /boot/limine.conf "/var/backups/limine.conf.$STAMP"
echo "backup: /var/backups/limine.conf.$STAMP"

if [[ ${1:-} == --uninstall ]]; then
  rm -f "$HOOK" /etc/limine-entry-tool.d/50-gpu-unlock.conf /var/lib/gpu-unlock-entry.stamp
  limine-entry-tool --remove-kernel linux-cachyos-gpu-unlock || true
  echo "Removed. /etc/gpu-unlock-entry.conf kept. Power off fully and boot the normal entry to relock the cards."
  exit 0
fi

# 1. stale hand-made entry
if grep -q '^/GPU1 BF16 trial (RTX 3090 only)' /boot/limine.conf; then
  limine-entry-tool --remove-entry "GPU1 BF16 trial (RTX 3090 only)"
fi
if [[ -d /boot/gpu-unlock-trial-20261031 ]]; then
  mv /boot/gpu-unlock-trial-20261031 "/var/backups/gpu-unlock-trial-20261031-stale-kernel-$STAMP"
fi

# 2. config + hook
[[ -f /etc/gpu-unlock-entry.conf ]] || cat >/etc/gpu-unlock-entry.conf <<'EOF'
# Used by /etc/boot/hooks/post.d/95-gpu-unlock-entry (see ~/ai/gpu-unlock/).
# Cards to unlock in the "linux-cachyos-gpu-unlock" boot entry: "all" or comma-separated PCI addresses.
KERNEL_PKG=linux-cachyos
TARGETS=0000:01:00.0,0000:02:00.0
EOF
install -Dm755 "$HERE/95-gpu-unlock-entry" "$HOOK"

# 3. regenerate the regular entry the normal way; the post hook adds/refreshes the unlock entry
limine-mkinitcpio linux-cachyos
echo
if grep -q 'linux-cachyos-gpu-unlock' /boot/limine.conf; then
  echo "OK: entry present:"
  awk '/linux-cachyos-gpu-unlock/{p=1} p{print} p&&/module_path/{exit}' /boot/limine.conf
else
  echo "ERROR: entry not created; check the messages above. Your normal entries are unchanged." >&2
  exit 1
fi
echo
limine-entry-tool --tree 3 || true
grep -E '^(default_entry|remember_last_entry|timeout)' /boot/limine.conf || true
echo
echo "Next: fully power OFF (not reboot), power on, pick 'linux-cachyos-gpu-unlock' under CachyOS."
