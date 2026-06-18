# Deploying on an Oracle Cloud VPS

The dashboard is a single Streamlit app with pre-computed data files committed to the
repo, so deployment is just: get the code, run it, expose it.

> **Oracle Cloud's #1 gotcha — there are TWO firewalls.** Opening a port in the OCI
> web console is not enough; the instance also ships with its own `iptables` rules that
> block everything except SSH. You must open the port in *both* places or the site will
> be unreachable even though the app is running. See step 4.

---

## Option A — Docker (recommended, reproducible)

```bash
# 1. Install Docker (Ubuntu)
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker

# 2. Get the code
git clone https://github.com/krish-mittal1/Traffic-management-Model.git
cd Traffic-management-Model

# 3. (optional) keep the MapMyIndia key out of the image
mkdir -p .streamlit
echo 'MAPPLS_KEY = "7748ecab0e753c215194a09e01debb77"' > .streamlit/secrets.toml

# 4. Build and run
docker compose up -d --build
```

The app now listens on `127.0.0.1:8501` inside the host (via the container's `8501`).
Put nginx in front (step 3 below) for port 80/443 + a domain.

Update later with: `git pull && docker compose up -d --build`

---

## Option B — systemd + venv (no Docker)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip nginx
git clone https://github.com/krish-mittal1/Traffic-management-Model.git
cd Traffic-management-Model
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # slim deps only

sudo cp deploy/parking.service /etc/systemd/system/parking.service
# edit User= and the paths in that file if your user isn't "ubuntu"
sudo systemctl daemon-reload
sudo systemctl enable --now parking
systemctl status parking                 # should be active (running)
```

Update later with: `git pull && sudo systemctl restart parking`

---

## 3. Reverse proxy with nginx (both options)

```bash
sudo apt install -y nginx
sudo cp deploy/nginx-parking.conf /etc/nginx/sites-available/parking
# edit server_name in that file to your domain (or leave _ to serve on the bare IP)
sudo ln -sf /etc/nginx/sites-available/parking /etc/nginx/sites-enabled/parking
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

HTTPS (only if you pointed a domain at the server):
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## 4. Open the firewall — BOTH layers

**Layer 1 — OCI console:** VCN → Security Lists (or the instance's Network Security
Group) → add Ingress rules: source `0.0.0.0/0`, TCP ports **80** and **443**.

**Layer 2 — on the instance:** Oracle images block these locally. On Ubuntu:
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```
(Oracle Linux uses firewalld instead: `sudo firewall-cmd --permanent --add-service=http
--add-service=https && sudo firewall-cmd --reload`.)

---

## 5. MapMyIndia key — whitelist the domain

In the [Mappls console](https://apis.mappls.com/console/) → your key → add your VPS
domain (or the public IP) to the allowed domains, or leave it unrestricted. Without this
the map tiles load locally but go blank when served from the VPS.

---

## Sanity checks

- App up locally on the box: `curl -I http://127.0.0.1:8501/_stcore/health` → `200`
- Through nginx: `curl -I http://<public-ip>/` → `200`
- Page hangs on "Connecting..." → the nginx `Upgrade`/`Connection` headers are missing.
- Page unreachable but app runs → firewall (step 4), almost always Layer 2.
- Maps blank only on the VPS → MapMyIndia domain whitelist (step 5).
