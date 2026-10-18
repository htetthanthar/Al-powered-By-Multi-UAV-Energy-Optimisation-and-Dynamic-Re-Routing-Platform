"""Verify TLS, authentication, events and Origin protection without exposing secrets."""
import base64
import json
from pathlib import Path
import urllib.request
import urllib.error

base = 'https://64.118.128.181'
password = Path('/opt/stratum/private/login.txt').read_text().split('Password: ')[1].strip()
auth = 'Basic ' + base64.b64encode(('stratum:' + password).encode()).decode()
def request(path, authenticated=True, data=None, origin=None, token=None):
    headers = {'Authorization': auth} if authenticated else {}
    if data is not None: headers['Content-Type'] = 'application/json'
    if origin: headers['Origin'] = origin
    if token: headers['X-UAV-Token'] = token
    req = urllib.request.Request(base+path, data=json.dumps(data).encode() if data is not None else None, headers=headers)
    try: return urllib.request.urlopen(req, timeout=15)
    except urllib.error.HTTPError as error: return error
for path in ['/', '/api/state', '/api/events', '/api/control']:
    with request(path, False) as r: assert r.status == 401, (path, r.status)
for path in ['/', '/style.css', '/app.js', '/cesium/Cesium.js']:
    with request(path) as r: assert r.status == 200, (path, r.status)
with request('/api/state') as r:
    state = json.load(r)
assert state['source'] == 'simulation'
with request('/api/events') as r: assert r.readline().startswith(b'data: ')
for origin, expected in [(base, 200), ('https://example.com', 403)]:
    with request('/api/control', data={'action': 'play' if state['running'] else 'pause'}, origin=origin, token=state['csrf']) as r:
        assert r.status == expected, (origin, r.status)
with request('/api/control', data={'action': 'pause'}, origin=base) as r: assert r.status == 403
print('PASS: trusted IP TLS, login protection, assets, live SSE, correct Origin and command-token enforcement')
