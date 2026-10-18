import time
import random
import asyncio
from enum import Enum
from typing import Dict, List, Optional

class RaftRole(Enum):
    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"
    LEADER = "LEADER"

class GCSConsensusNode:
    def __init__(self, node_id: str, cluster_nodes: List[str]):
        self.node_id = node_id
        self.cluster = cluster_nodes
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.role = RaftRole.FOLLOWER
        self.log: List[Dict] = []
        self.commit_index = 0
        
        # Timing Parameters
        self.last_heartbeat_time = time.time()
        self.election_timeout_s = random.uniform(0.15, 0.30)
        self.heartbeat_interval_s = 0.05

    def handle_heartbeat_tick(self) -> Optional[Dict]:
        """
        Evaluates election timeout in Follower/Candidate state, or sends heartbeats if Leader.
        """
        now = time.time()
        
        if self.role == RaftRole.LEADER:
            if now - self.last_heartbeat_time >= self.heartbeat_interval_s:
                self.last_heartbeat_time = now
                return {
                    "type": "APPEND_ENTRIES_HEARTBEAT",
                    "term": self.current_term,
                    "leader_id": self.node_id,
                    "commit_index": self.commit_index
                }
        else:
            if now - self.last_heartbeat_time >= self.election_timeout_s:
                # Transition to Candidate and initiate vote
                self.role = RaftRole.CANDIDATE
                self.current_term += 1
                self.voted_for = self.node_id
                self.last_heartbeat_time = now
                self.election_timeout_s = random.uniform(0.15, 0.30)
                
                return {
                    "type": "REQUEST_VOTE",
                    "term": self.current_term,
                    "candidate_id": self.node_id,
                    "last_log_idx": len(self.log) - 1
                }
        return None

    def receive_request_vote(self, candidate_term: int, candidate_id: str) -> bool:
        """Processes incoming RequestVote RPC."""
        if candidate_term > self.current_term:
            self.current_term = candidate_term
            self.role = RaftRole.FOLLOWER
            self.voted_for = None

        if candidate_term == self.current_term and (self.voted_for is None or self.voted_for == candidate_id):
            self.voted_for = candidate_id
            self.last_heartbeat_time = time.time()
            return True
        return False

    def receive_heartbeat(self, leader_term: int, leader_id: str):
        """Processes incoming AppendEntries heartbeat."""
        if leader_term >= self.current_term:
            self.current_term = leader_term
            self.role = RaftRole.FOLLOWER
            self.last_heartbeat_time = time.time()

    def append_mission_command(self, uav_id: str, action: str, params: Dict) -> bool:
        """Appends new command entry to state log if Leader."""
        if self.role != RaftRole.LEADER:
            return False
        
        entry = {
            "index": len(self.log),
            "term": self.current_term,
            "uav_id": uav_id,
            "action": action,
            "params": params,
            "timestamp": time.time()
        }
        self.log.append(entry)
        self.commit_index = len(self.log) - 1
        return True

async def simulate_raft_gcs_cluster():
    cluster_ids = ["GCS-ALPHA", "GCS-BETA", "GCS-GAMMA"]
    nodes = {nid: GCSConsensusNode(nid, cluster_ids) for nid in cluster_ids}

    print("==========================================================================")
    print("      TRIPLE-REDUNDANT GCS RAFT CONSENSUS & FAILOVER SIMULATION           ")
    print("==========================================================================")

    # Force GCS-ALPHA to initiate election first
    nodes["GCS-ALPHA"].election_timeout_s = 0.05
    await asyncio.sleep(0.08)

    # 1. Trigger Election
    vote_req = nodes["GCS-ALPHA"].handle_heartbeat_tick()
    if vote_req and vote_req["type"] == "REQUEST_VOTE":
        votes = 1 # Self vote
        for nid in ["GCS-BETA", "GCS-GAMMA"]:
            if nodes[nid].receive_request_vote(vote_req["term"], "GCS-ALPHA"):
                votes += 1
        if votes >= 2: # Quorum achieved
            nodes["GCS-ALPHA"].role = RaftRole.LEADER
            print(f"[+] Leader Elected: 'GCS-ALPHA' (Term: {nodes['GCS-ALPHA'].current_term}, Votes: {votes}/3)")

    # 2. Leader Replicates Mission Command Log
    success = nodes["GCS-ALPHA"].append_mission_command(
        uav_id="UAV-01", action="DISPATCH_TRAJECTORY", params={"route_id": "RT-SG-001"}
    )
    print(f"[*] Mission Command Appended to Leader Log: Success={success} | Total Log Entries: {len(nodes['GCS-ALPHA'].log)}")

    # 3. Simulate Sudden Hardware Failure of Primary Leader (GCS-ALPHA crashes)
    print("\n[!] CRITICAL EVENT: Primary Leader 'GCS-ALPHA' hardware server failure (Loss of Heartbeat)...")
    del nodes["GCS-ALPHA"]

    # Beta election timeout fires
    nodes["GCS-BETA"].election_timeout_s = 0.10
    nodes["GCS-GAMMA"].election_timeout_s = 0.25
    await asyncio.sleep(0.12)

    vote_req_beta = nodes["GCS-BETA"].handle_heartbeat_tick()
    if vote_req_beta and vote_req_beta["type"] == "REQUEST_VOTE":
        votes_beta = 1
        if nodes["GCS-GAMMA"].receive_request_vote(vote_req_beta["term"], "GCS-BETA"):
            votes_beta += 1
        if votes_beta >= 2:
            nodes["GCS-BETA"].role = RaftRole.LEADER
            print(f"[+] Failover Successful! New Leader Elected: 'GCS-BETA' (Term: {nodes['GCS-BETA'].current_term}, Votes: {votes_beta}/2)")

    # Verify cluster health
    print("\nActive Cluster Node State:")
    for nid, node in nodes.items():
        print(f"  • {nid:10s} : Role={node.role.value:10s} | Term={node.current_term}")
    print("==========================================================================\n")

if __name__ == "__main__":
    asyncio.run(simulate_raft_gcs_cluster())