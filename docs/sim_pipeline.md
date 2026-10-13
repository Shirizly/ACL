# Simulator Pipeline Commands

## Start stack

```bash
scripts/sim/up.sh
scripts/sim/start_xarm7.sh
```

## Check status

```bash
scripts/sim/status.sh
```

## Access web simulation

Open `http://127.0.0.1:18333` in your browser.

## Enter simulator shell

```bash
scripts/sim/shell.sh
```

## Stop stack

```bash
scripts/sim/down.sh
```

## Equivalent manual commands

```bash
docker compose -f docker-compose.sim.yml up -d uf_sim
docker exec -it uf_software /xarm_scripts/xarm_start.sh 7 7
```
