"""Create a distinct dashboard credential without displaying it or replacing one."""
import grp
import os
from pathlib import Path
import secrets
import subprocess

os.umask(0o077)
target = Path('/etc/nginx/stratum.htpasswd')
if target.exists():
    raise SystemExit('Existing dashboard credential found; refusing overwrite')
private = Path('/opt/stratum/private')
private.mkdir(mode=0o700, exist_ok=True)
password = secrets.token_urlsafe(24)
hashed = subprocess.run(['openssl', 'passwd', '-6', '-stdin'], input=password+'\n', text=True, capture_output=True, check=True).stdout.strip()
with (private / 'login.txt').open('x') as out:
    out.write('URL: https://64.118.128.181/\nUsername: stratum\nPassword: '+password+'\n')
with target.open('x') as out:
    out.write('stratum:'+hashed+'\n')
os.chown(target, 0, grp.getgrnam('www-data').gr_gid)
target.chmod(0o640)
print('Dashboard credential stored in root-only /opt/stratum/private/login.txt')
