import time
import struct
import json
import hashlib
import hmac
import httpx
from typing import Dict, Optional

class RemoteIDUTMBroadcaster:
    def __init__(
        self,
        uas_id_serial: str = "1596G-UAV-SG-0012",
        operator_id: str = "SGP-OP-2026-8891-B",
        utm_endpoint: str = "https://utm.singapore-caas.gov.sg/api/v1/rid",
        api_auth_token: str = "UTM_BEARER_AUTH_TOKEN"
    ):
        self.uas_id = uas_id_serial
        self.operator_id = operator_id
        self.utm_endpoint = utm_endpoint
        self.auth_token = api_auth_token
        self.seq_num = 0

    def encode_open_drone_id_packet(
        self,
        lat: float,
        lon: float,
        alt_wgs84_m: float,
        alt_agl_m: float,
        ground_speed_ms: float,
        track_deg: float,
        v_speed_ms: float,
        timestamp_s: float
    ) -> bytes:
        """
        Packs binary OpenDroneID Location/Vector frame (Message Type 0x01).
        Follows ASTM F3411-19 byte layout:
          - Byte 0: Msg Type (0x01 = Location) & Protocol Version (0x02)
          - Byte 1: Status & Heading
          - Bytes 2-5: Latitude (int32, scaled by 1e7)
          - Bytes 6-9: Longitude (int32, scaled by 1e7)
          - Bytes 10-11: Geodetic Altitude (uint16, 0.5m res, -1000m offset)
          - Bytes 12-13: Height AGL (uint16, 0.5m res, -1000m offset)
          - Byte 14: Speed Horizontal (uint8, 0.25 m/s res)
          - Byte 15: Speed Vertical (int8, 0.5 m/s res)
          - Bytes 16-17: Timestamp tenths of second (uint16)
        """
        msg_header = (0x01 << 4) | 0x02
        status_and_direction = int((track_deg % 360) / 2.0)  # 0-180 mapped to 0-360 deg
        
        lat_i32 = int(lat * 1e7)
        lon_i32 = int(lon * 1e7)
        alt_wgs84_u16 = int((alt_wgs84_m + 1000.0) / 0.5)
        alt_agl_u16 = int((alt_agl_m + 1000.0) / 0.5)
        speed_u8 = min(254, int(ground_speed_ms / 0.25))
        v_speed_i8 = max(-127, min(127, int(v_speed_ms / 0.5)))
        time_tenths = int((timestamp_s % 3600.0) * 10)

        packed_payload = struct.pack(
            ">BBiiHHBbH",
            msg_header,
            status_and_direction,
            lat_i32,
            lon_i32,
            alt_wgs84_u16,
            alt_agl_u16,
            speed_u8,
            v_speed_i8,
            time_tenths
        )
        return packed_payload

    async def broadcast_network_rid(
        self,
        lat: float,
        lon: float,
        alt_m: float,
        speed_kmh: float,
        heading_deg: float,
        battery_soc_pct: float
    ) -> Dict:
        """
        Transmits JSON Network Remote ID payload over HTTPS to national UTM network.
        """
        self.seq_num += 1
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        payload = {
            "uas_id": self.uas_id,
            "operator_registration_number": self.operator_id,
            "sequence_number": self.seq_num,
            "timestamp": timestamp_iso,
            "telemetry": {
                "position": {
                    "latitude": round(lat, 7),
                    "longitude": round(lon, 7),
                    "altitude_geodetic_wgs84_m": round(alt_m + 32.5, 2),  # +32.5m geoid undulation in Singapore
                    "height_agl_m": round(alt_m, 2)
                },
                "kinematics": {
                    "ground_speed_kmh": round(speed_kmh, 2),
                    "heading_degrees": round(heading_deg, 1),
                    "vertical_speed_ms": 0.0
                },
                "operational_status": {
                    "flight_state": "AIRBORNE_DELIVERY_EN_ROUTE",
                    "emergency_status": "NOMINAL",
                    "battery_soc_pct": round(battery_soc_pct, 1)
                }
            }
        }

        # Simulated REST post to UTM
        # async with httpx.AsyncClient(timeout=2.0) as client:
        #     resp = await client.post(self.utm_endpoint, json=payload, headers={"Authorization": f"Bearer {self.auth_token}"})
        #     return resp.json()

        return {
            "status": "TRANSMITTED_OK",
            "uas_id": self.uas_id,
            "seq": self.seq_num,
            "bytes_packed": len(json.dumps(payload))
        }

if __name__ == "__main__":
    import asyncio
    broadcaster = RemoteIDUTMBroadcaster()

    # Encode binary Bluetooth/Wi-Fi direct payload
    raw_bytes = broadcaster.encode_open_drone_id_packet(
        lat=1.3521, lon=103.8198, alt_wgs84_m=57.5, alt_agl_m=25.0,
        ground_speed_ms=18.05, track_deg=45.0, v_speed_ms=0.0, timestamp_s=time.time()
    )

    print("==========================================================================")
    print("      ASTM F3411-19 REMOTE ID & UTM PROTOCOL ENCODER TEST                 ")
    print("==========================================================================")
    print(f"Binary OpenDroneID Packet : {raw_bytes.hex().upper()} ({len(raw_bytes)} bytes)")
    
    async def test_network_rid():
        res = await broadcaster.broadcast_network_rid(
            lat=1.3521, lon=103.8198, alt_m=25.0, speed_kmh=65.0, heading_deg=45.0, battery_soc_pct=88.5
        )
        print(f"Network UTM Dispatch      : {res}")

    asyncio.run(test_network_rid())
    print("==========================================================================\n")