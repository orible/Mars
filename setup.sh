#!/usr/bin/env bash
# Sets up the Mars repo on macOS or Linux: checks Go and the .NET SDK,
# downloads Go modules, builds the simulator tools and the observer, and
# creates the training folders. Safe to run again; finished steps are
# skipped.
#
#   ./setup.sh             check, build
#   ./setup.sh --install   also install a missing Go / .NET SDK (Homebrew on macOS)
#   ./setup.sh --test      also run the Go tests
#   ./setup.sh --mocap     also download the CMU motion capture (about 12 MB) used
#                          to re-extract the get-up references (already committed)
#   ./setup.sh --save      snapshot training data (sim/checkpoints, sim/runs) into
#                          training/, which git carries to other systems
#   ./setup.sh --restore   replace local training data with the training/ snapshot
#                          (done automatically when there is no local checkpoint)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
min_go_major=1
min_go_minor=22
dotnet_major=10
install=0 test=0 mocap=0 save=0 restore=0
for arg in "$@"; do
  case "$arg" in
    --install) install=1 ;;
    --test) test=1 ;;
    --mocap) mocap=1 ;;
    --save) save=1 ;;
    --restore) restore=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

step() { printf '\033[36m==> %s\033[0m\n' "$*"; }
ok() { printf '\033[32m    %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    %s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# --- Training data snapshot -------------------------------------------------
# The trainer rewrites sim/checkpoints and sim/runs every iteration, so
# those stay out of git; training/ holds a snapshot that git carries.
snap="$root/training"
copy_tree() { # from to → prints the number of files copied
  [ -d "$1" ] || { echo 0; return; }
  mkdir -p "$2"
  local n=0 f
  for f in "$1"/*; do [ -f "$f" ] && cp -f "$f" "$2/" && n=$((n+1)); done
  echo "$n"
}
if [ "$save" = 1 ]; then
  step "Saving the training data snapshot (training/)"
  n=$(( $(copy_tree "$root/sim/checkpoints" "$snap/checkpoints") + $(copy_tree "$root/sim/runs" "$snap/runs") ))
  if [ "$n" = 0 ]; then warn "no training data in sim/checkpoints or sim/runs yet"
  else ok "$n files; commit training/ to carry them: git add training && git commit -m 'training snapshot'"; fi
  exit 0
fi

# Common install locations that may not be on PATH yet.
export PATH="$PATH:/usr/local/go/bin:$HOME/go/bin:$HOME/.dotnet"

go_ok() {
  command -v go >/dev/null 2>&1 || return 1
  local v; v="$(go version | sed -nE 's/.*go([0-9]+)\.([0-9]+).*/\1 \2/p')"
  [ -n "$v" ] || return 1
  set -- $v
  [ "$1" -gt "$min_go_major" ] || { [ "$1" -eq "$min_go_major" ] && [ "$2" -ge "$min_go_minor" ]; }
}

dotnet_ok() {
  command -v dotnet >/dev/null 2>&1 && dotnet --list-sdks 2>/dev/null | grep -q "^${dotnet_major}\."
}

brew_install() {
  command -v brew >/dev/null 2>&1 || return 1
  step "Installing $1 with Homebrew"
  brew install "$2"
}

# --- Go -------------------------------------------------------------------
step "Go ${min_go_major}.${min_go_minor} or newer"
if go_ok; then
  ok "$(go version)"
elif [ "$install" = 1 ] && brew_install Go go && go_ok; then
  ok "$(go version)"
else
  command -v go >/dev/null 2>&1 && warn "found $(go version), need ${min_go_major}.${min_go_minor}+" || warn "Go not found"
  die "Install Go ${min_go_major}.${min_go_minor}+ from https://go.dev/dl/ (distribution packages are often too old)$( [ "$install" = 1 ] || echo ', or on macOS run: ./setup.sh --install')"
fi

# --- .NET SDK ---------------------------------------------------------------
step ".NET SDK ${dotnet_major}"
if dotnet_ok; then
  ok "$(dotnet --list-sdks | grep "^${dotnet_major}\." | head -n1)"
elif [ "$install" = 1 ] && brew_install ".NET SDK" dotnet-sdk && dotnet_ok; then
  ok "installed"
else
  warn ".NET SDK ${dotnet_major} not found"
  die "Install the .NET ${dotnet_major} SDK: https://dotnet.microsoft.com/download (Linux: your distribution's dotnet-sdk-${dotnet_major}.0 package or Microsoft's packages)$( [ "$install" = 1 ] || echo ', or on macOS run: ./setup.sh --install')"
fi

# --- Simulator ---------------------------------------------------------------
cd "$root/sim"
step "Go modules"
go mod download
ok "downloaded"

step "Building simulator tools (sim/bin)"
mkdir -p bin
for tool in simserver train mocapref bodygen; do
  go build -o "bin/$tool" "./cmd/$tool"
  ok "bin/$tool"
done

step "Training folders"
mkdir -p checkpoints runs
ok "sim/checkpoints, sim/runs"
if [ -d "$snap" ]; then
  if [ "$restore" = 1 ] || [ ! -f checkpoints/policy.json ]; then
    step "Restoring training data from training/"
    n=$(( $(copy_tree "$snap/checkpoints" checkpoints) + $(copy_tree "$snap/runs" runs) ))
    ok "$n files; resume with: cd sim && ./bin/train -scenario getup -resume"
  else
    ok "local training data kept (use --restore to replace it with the training/ snapshot)"
  fi
fi

if [ "$test" = 1 ]; then
  step "Go tests"
  go test ./...
fi

# --- Observer ----------------------------------------------------------------
step "Building the observer (Blazor)"
dotnet build "$root/observer" --nologo -v q
ok "built"

# --- Motion capture (optional) ------------------------------------------------
if [ "$mocap" = 1 ]; then
  step "CMU motion capture (http://mocap.cs.cmu.edu; free for research and commercial use per CMU)"
  fetch() { # subject file
    local dir="$root/sim/mocap/cmu-$1"
    mkdir -p "$dir"
    [ -f "$dir/$2" ] && return 0
    curl -fsSL --max-time 120 -o "$dir/$2" "http://mocap.cs.cmu.edu/subjects/$1/$2"
    ok "mocap/cmu-$1/$2"
  }
  for f in 114.asf 114_02.amc 114_11.amc; do fetch 114 "$f"; done
  for f in 139.asf 139_16.amc 139_17.amc 139_18.amc; do fetch 139 "$f"; done
  for f in 140.asf 140_01.amc 140_02.amc 140_03.amc 140_04.amc 140_08.amc 140_09.amc; do fetch 140 "$f"; done
  ok "all recordings present"
fi

echo
printf '\033[36mReady. Next:\033[0m\n'
echo "  cd sim && ./bin/train -scenario getup                   # train (writes checkpoints/policy.json)"
echo "  cd sim && ./bin/simserver -policy checkpoints/policy.json -follow"
echo "  dotnet run --project observer                           # then open http://localhost:5217"
