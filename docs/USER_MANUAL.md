# User manual: preprocess a trajectory, run it in the simulator, run it on the robot

Every command is run from the repository root, in a terminal where the virtual environment is active:

```bash
cd ~/Code/ACL                 # wherever the repository is checked out
source .venv/bin/activate     # the scripts call `python`, so the venv must be active
```

`XARM_IP` selects the robot: unset (or `127.0.0.1`) means the simulator; set it to the real arm's IP for the robot.

Background reading, if needed: `docs/trajectory_format.md` (CSV format) and `docs/safety_walls.md` (how the safety
layers work).

---

## 0. One-time setup (new machine)

### 0.1 What you need

| | Needed for | Notes |
|---|---|---|
| Linux on an **x86-64 (Intel/AMD)** PC | everything | tested on Pop!_OS / Ubuntu 22.04. The simulator image is x86-64 only, so it won't run on ARM machines (e.g. Apple Silicon). |
| Python **3.11 or newer** | everything | tested with 3.13 |
| git | getting the code | |
| Docker Engine + the Compose plugin | **simulator only** | not needed to drive the real robot |
| ~3 GB free disk | | simulator image 1.2 GB, browser for recording ~0.5 GB |
| Ethernet to the robot controller | **real robot only** | see 0.6 |

### 0.2 Get the code

```bash
git clone https://github.com/Shirizly/ACL.git
cd ACL
```

`data/` (including the preprocessed sets) is in git. `outputs/` (reports, videos) is not; it is created locally.

### 0.3 Python environment

```bash
sudo apt install -y python3 python3-venv python3-pip       # Ubuntu/Pop!_OS; skip if already installed
python3 --version                                          # must be 3.11 or newer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install --with-deps chromium          # browser for video recording + its system libraries (asks for sudo)
python -m pytest -q tests                                  # should end with "... passed"; no robot or simulator needed
```

* Without sudo, use `python -m playwright install chromium` instead. If recording later fails with missing
  libraries, ask an admin to run `python -m playwright install-deps chromium`.
* Every new terminal needs `cd` into the repository and `source .venv/bin/activate` again.

### 0.4 Docker (simulator only)

Install Docker Engine with the Compose plugin. Follow https://docs.docker.com/engine/install/ for your
distribution, or use Docker's convenience script:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER        # run docker without sudo...
newgrp docker                        # ...takes effect after logging out and in; newgrp applies it to this terminal
docker run --rm hello-world          # should print "Hello from Docker!"
docker compose version               # should print a version (the Compose plugin)
```

### 0.5 Simulator: first start

```bash
docker compose -f docker-compose.sim.yml pull          # downloads the simulator image (1.2 GB, once)
scripts/sim/up.sh                                      # creates and starts the container "uf_software"
docker exec -d uf_software /xarm_scripts/xarm_start.sh 7 7   # starts the simulated xArm 7 controller + Studio
sleep 30
scripts/control/measure.sh where                       # must print joint angles: the SDK can talk to the simulator
```

* The simulator uses host ports **18333** (Studio web page), **502–504** and **30000–30003** (robot protocol).
  If `up.sh` complains that a port is in use, find the program holding it with
  `ss -ltnp | grep -E ':(18333|502|503|504|3000[0-3]) '` and stop it.
* Open **http://127.0.0.1:18333**. If Studio shows a serial-number (SN) warning, close it; it doesn't matter in
  the simulator.
* Tell the simulator which gripper is mounted. Studio then draws it, and the controller's own self-collision
  check accounts for it:
  ```bash
  python -c "from xarm.wrapper import XArmAPI; a = XArmAPI('127.0.0.1'); print('rc', a.set_collision_tool_model(1)); a.disconnect()"
  ```
  `1` = xArm Gripper. Other values are listed in the SDK's `set_collision_tool_model`.
* `scripts/sim/down.sh` removes the container, and with it these settings. After the next `up.sh`, repeat the
  `xarm_start.sh` and tool-model commands.
* Then put the arm in the ready pose: `scripts/control/go_ready.sh --force --yes` (see 3a).

### 0.6 Real robot: network

1. Connect the PC to the robot **controller box** by Ethernet. Give the PC a static IP in the robot's subnet,
   e.g. robot `192.168.1.208` → PC `192.168.1.100`, netmask `255.255.255.0`, no gateway.
2. Check:
   ```bash
   export XARM_IP=192.168.1.208           # the robot's IP (printed on the controller / shown in Studio)
   ping -c 3 $XARM_IP
   scripts/control/measure.sh where       # prints the real arm's joints; does not move it
   ```
3. Studio for the real arm is at **http://$XARM_IP:18333** in a browser.

The Python SDK is installed by `pip install -r requirements.txt`. No Docker is needed for the real robot.

### 0.7 Check the safety configuration matches this setup

`config/safety.yaml` is in git, so a fresh clone has the configuration last committed. Check that it describes
the cell you are working in (section 1 and section 4a), and that its preprocessed set is found:

```bash
python -m src.xarm_pipeline.safety.preprocessed list    # "<- current config/safety.yaml" marks the matching set
```

If no set is marked, the configuration differs from the one used to preprocess. Run the preprocessing again
(section 2) before playing anything.

*(The repository also has a VS Code Dev Container setup in `.devcontainer/`. It has not been tested with this
pipeline; the venv on the host described above is what was used and verified.)*

---

## 1. The safety configuration (check before anything else)

Everything (preprocessing, checks, playback) uses `config/safety.yaml`: the workcell walls, the gripper geometry,
the margins. Preprocessed trajectories are stored **per configuration**. If you change the walls or the gripper
in that file, earlier preprocessed outputs no longer apply, and preprocessing must be run again; it creates a new
folder automatically.

```bash
python -m src.xarm_pipeline.safety.preprocessed list    # preprocessed sets; the one for the current config is marked
```

---

## 2. Preprocess a trajectory

### 2a. A recorded (motion-capture) trajectory

Choose the folder, the **time scale** (2.5 = 2.5× slower than recorded), and the trajectory number(s):

```bash
scripts/control/prepare_recorded.sh data/recorded_trajectories/5ec359d2 2.5 18
```

It prints what it did, for example:

```
0018.csv: time scale x2.5, duration 31.1 s (limits at 90% of the robot's)
  nothing cut
  part 0018__p1: 8.0 s (original t 0.00-3.15 s), yaw 0 deg, slack 30 mm, +0.15 s to start/stop at rest
  part 0018__p2: ...
  kept 100% of the time; time scale needed for no speed cuts: >= 1.92
  wrote data/preprocessed/001_D1020xW850xH1180/recorded/5ec359d2/0018_t2.50.csv (+ .fit.json, .prep.json)
```

* **`NOTICE - cut for ...`** lines list parts that were removed because they were too fast or didn't fit. To
  keep more, use a larger time scale: the last line tells you the scale needed for no speed cuts.
* The trajectory may be split into **parts** (`0018__p1`, `__p2`, ...). Each part is placed in the cell on its
  own, and each starts and ends at rest.
* Several at once: `scripts/control/prepare_recorded.sh data/recorded_trajectories/5ec359d2 2.5 18 9 19`.
* It takes about 1–3 minutes per trajectory. No robot is needed.

The output path (`data/preprocessed/<set>/recorded/...csv`) is what you play in sections 3 and 4. Below,
**`TRAJ`** stands for it:

```bash
TRAJ=data/preprocessed/001_D1020xW850xH1180/recorded/5ec359d2/0018_t2.50.csv
python -m src.xarm_pipeline.trajectory_io "$TRAJ"        # lists its parts, durations, and how to name one
```

### 2b. Any other Cartesian trajectory (CSV in the format of `docs/trajectory_format.md`)

Nothing needs to be done ahead of time. Playing the original file fits it into the cell automatically on first
use (section 3). To prepare it in advance:

```bash
python -m src.xarm_pipeline.validate_trajectory data/trajectories/my_traj.csv     # format check
scripts/control/fit_trajectories.sh data/trajectories/my_traj.csv                  # fit into the cell
```

### 2c. The built-in test trajectories (for the first real-robot session)

```bash
python -m src.xarm_pipeline.safety.test_trajectories corners    # tour of the 8 cell corners, slow (~4 min)
python -m src.xarm_pipeline.safety.test_trajectories dynamic    # short strokes at 20/50/100% of the robot limits
```

They are written to `data/preprocessed/<set>/tests/`.

---

## 3. Run it in the simulator

### 3a. Start the simulator (once per session)

```bash
scripts/sim/up.sh                                          # starts the simulator container
docker exec -d uf_software /xarm_scripts/xarm_start.sh 7 7  # starts the simulated xArm 7 controller
sleep 30                                                   # it needs ~20-40 s to come up
python -c "from xarm.wrapper import XArmAPI; a = XArmAPI('127.0.0.1'); print('rc', a.set_collision_tool_model(1)); a.disconnect()"   # gripper model (see 0.5)
scripts/control/go_ready.sh --force --yes                  # arm to the ready pose
```

`--force` is only for the simulator: after a start the arm is folded, with the gripper model "below the floor",
and the normal safety check would refuse to move it.

If a later command fails with `failed to check version`, the simulated controller has hung. This happens after
many runs. Restart it:

```bash
docker restart uf_software && docker exec -d uf_software /xarm_scripts/xarm_start.sh 7 7 && sleep 30
scripts/control/go_ready.sh --force --yes
```

### 3b. Watch it live

Open **http://127.0.0.1:18333** in a browser (UFACTORY Studio's 3D view of the arm). To see the arm with the cell
walls and the gripper model drawn in, filling the window:

```bash
python -m src.xarm_pipeline.sim_scene_overlay --arm-only     # Ctrl+C closes it
```

### 3c. Play the trajectory

```bash
scripts/control/play_cartesian_ik.sh "$TRAJ" servo           # all parts, in order
scripts/control/play_cartesian_ik.sh "$TRAJ#0018__p2" servo  # one part only
```

What happens:
1. A safety check of the whole part from the arm's current state.
2. A joint move to the part's start.
3. Playback, with the runtime monitor and the robot's boundary active.

Repeated for each part.

Exit code / last line:

| Result | Meaning |
|---|---|
| exit 0 | completed |
| exit 2, `REJECTED ... the robot was not moved` | the safety check refused it; the message says which part of the arm and which wall |
| exit 3, `SAFETY STOP` / `ABORTED` | stopped during the run (runtime monitor or robot firmware); the message gives the reason, e.g. `C22 Self-Collision Error` |

After a stop, `scripts/control/go_ready.sh --yes` clears the error and returns to the ready pose. In the sim, use
`--force` if the arm stopped in an awkward spot.

Test trajectories:

```bash
SET=$(ls -d data/preprocessed/001_*)        # the set folder for your config (see section 1)
scripts/control/play_cartesian_ik.sh "$SET/tests/dynamic_tests.csv#dyn_y_20pct" servo
python -m src.xarm_pipeline.playback --trajectory "$SET/tests/workspace_corners.csv#corner_tour" --control servo
```

The corner tour is a **joint** trajectory, so it uses `playback` with `--control servo` instead of `play_cartesian_ik.sh`.

### 3d. Get a video (and a tracking-error report)

```bash
XARM_RECORD_ARGS="--headless --arm-only --walls" scripts/control/play_cartesian_ik_record.sh "$TRAJ" servo 120
```

* `120` is the maximum recording length in seconds. Recording stops by itself 1.5 s after the trajectory ends.
* Leave out `XARM_RECORD_ARGS` to watch the full Studio page in a visible browser while it records. Without
  `--walls` the cell isn't drawn.
* Results go to **`outputs/reports/<date>_<time>_servo/`**, which contains:
  * `*_ufactory_studio.webm`: the video (opens in a browser or VLC);
  * `cartesian_tracking_error.png`, `cartesian_error_metrics.json`: commanded vs measured position;
  * `safety_*`: safety check report and monitor log (closest approach to each wall);
  * `part1/`, `part2/`, ...: per part, when the trajectory has several.

Video of the joint corner tour:

```bash
python -m src.xarm_pipeline.record_web_sim --headless --arm-only --walls --seconds 300 --stop-when-done 1.5 \
  --output-dir outputs/videos \
  --run-command "python -m src.xarm_pipeline.playback --trajectory '$SET/tests/workspace_corners.csv#corner_tour' --control servo"
```

### 3e. Stop the simulator

```bash
scripts/sim/down.sh
```

---

## 4. Run the same on the real robot

The commands are the same as in the simulator, with `XARM_IP` set to the robot. Run nothing on the real robot
that hasn't run cleanly in the simulator with the same config.

### 4a. Before the first run (once per setup)

1. **Measured config.** `config/safety.yaml` describes the real cell: walls (`measure.sh walls` can measure them
   by touching each wall), gripper box, margin.
2. **TCP offset.** `geometry.firmware_tcp_offset` equals the TCP offset set in Studio. Playback reads the robot's
   value and prints a warning if they differ.
3. **Studio settings.** Set the end effector to the actual gripper, and turn **self-collision detection on**. It
   catches arm-on-arm contact that the wall checks don't model.
4. **Gripper model.** Check it against the real gripper, without moving the arm:
   ```bash
   export XARM_IP=192.168.1.XXX
   scripts/control/measure.sh where        # model's gripper/tip position vs a ruler, plus clearance to each wall
   python -m src.xarm_pipeline.sim_scene_overlay --url http://$XARM_IP:18333 --arm-only   # walls + gripper on the live arm
   ```
5. **Stop test against a virtual wall**, in free space, slow, asks before moving:
   `scripts/control/measure.sh wall-test --direction y+`.
6. **Measure stopping time.** Set `runtime.latency_s` and `runtime.stopping_time_s` from a real stop
   (see `docs/safety_walls.md`). They decide how early the monitor stops a fast motion.

### 4b. Every session

```bash
export XARM_IP=192.168.1.XXX              # the robot's IP
```

* Cell clear of people, emergency stop in reach, robot enabled in Studio with no errors.
* `scripts/control/go_ready.sh` checks the move, asks, and moves to the ready pose. Never use `--force` on the
  real robot.

### 4c. Suggested order for the first real tests

```bash
SET=$(ls -d data/preprocessed/001_*)
python -m src.xarm_pipeline.playback --trajectory "$SET/tests/workspace_corners.csv#corner_tour" --control servo
scripts/control/play_cartesian_ik.sh "$SET/tests/dynamic_tests.csv#dyn_x_20pct" servo   # then y, z
scripts/control/play_cartesian_ik.sh "$SET/tests/dynamic_tests.csv#dyn_x_50pct" servo   # then 100% only if 50% was clean
```

For recorded trajectories, start with a **generous time scale** (e.g. 4) and lower it step by step toward the
value the preprocessing reported:

```bash
scripts/control/prepare_recorded.sh data/recorded_trajectories/5ec359d2 4 18
scripts/control/play_cartesian_ik.sh "data/preprocessed/001_D1020xW850xH1180/recorded/5ec359d2/0018_t4.00.csv" servo
```

### 4d. Record a real run

The recorder captures Studio's 3D view, which on the real robot shows the arm's live digital twin. It doesn't
record a camera; film the robot separately if you need real footage. The tracking-error report is measured
from the real arm.

```bash
SIM_WEB_URL=http://$XARM_IP:18333 XARM_RECORD_ARGS="--headless --arm-only --walls" \
  scripts/control/play_cartesian_ik_record.sh "$TRAJ" servo 120
```

Recording costs CPU and has caused command-timing hiccups. For fast trajectories, prefer to run without
recording, and record a slower version for viewing.

### 4e. If something goes wrong

| Situation | Do |
|---|---|
| Anything unexpected | press the emergency stop |
| `SAFETY STOP` / exit 3 | read the reason on the last lines; clear with `scripts/control/go_ready.sh` (checks first), or jog in Studio |
| `REJECTED` / exit 2 | nothing moved; the message names the arm part and the wall. Larger time scale, or check the config |
| Robot still limited by a boundary after an aborted run | `scripts/control/firmware_fence.sh status`, then `scripts/control/firmware_fence.sh reset` |
| `PRESTART MISMATCH` | the robot's TCP offset differs from `geometry.firmware_tcp_offset` in the config |

---

## Quick reference

| Task | Command |
|---|---|
| Preprocess a recording | `scripts/control/prepare_recorded.sh <folder> <time scale> <number>` |
| List the parts of a file | `python -m src.xarm_pipeline.trajectory_io <file>` |
| First-time setup | section 0: venv + `pip install -r requirements.txt` + `python -m playwright install --with-deps chromium`; Docker for the sim |
| Start sim / controller | `scripts/sim/up.sh` ; `docker exec -d uf_software /xarm_scripts/xarm_start.sh 7 7` (+ gripper model, 3a) |
| Ready pose | `scripts/control/go_ready.sh` (sim after a start: `--force --yes`) |
| Watch live | browser at `http://<ip>:18333`, or `python -m src.xarm_pipeline.sim_scene_overlay --arm-only [--url ...]` |
| Play (Cartesian) | `scripts/control/play_cartesian_ik.sh "<file>[#part]" servo` |
| Play (joint, e.g. corner tour) | `python -m src.xarm_pipeline.playback --trajectory "<file>#corner_tour" --control servo` |
| Play + video + report | `XARM_RECORD_ARGS="--headless --arm-only --walls" scripts/control/play_cartesian_ik_record.sh "<file>" servo <max s>` |
| Where is the gripper (no motion) | `scripts/control/measure.sh where` |
| Stop the sim | `scripts/sim/down.sh` |
