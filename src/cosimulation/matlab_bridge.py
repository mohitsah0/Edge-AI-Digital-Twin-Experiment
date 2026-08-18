"""
MATLAB/Simulink ↔ Python Co-simulation Bridge
===============================================
Connects the Simulink microgrid model to the Python Edge-AI stack via
a lightweight TCP/IP socket server.

┌─────────────────────────────────────────────────────────────┐
│  MATLAB / Simulink                                          │
│  Microgrid model  ──TCP send──►  CoSimBridge (this file)   │
│        ▲                                   │               │
│        └────── TCP recv setpoints ─────────┘               │
└─────────────────────────────────────────────────────────────┘

Message protocol (JSON, newline-delimited):
  MATLAB → Python : {"v": 480.5, "f": 60.01, "p": 30000, "q": 1000,
                      "soc": 65.0, "irr": 800.0, "t": 1721000000.0}
  Python → MATLAB : {"pv_kw": 40.0, "bess_kw": 0.0, "diesel_kw": 40.0,
                      "v_ref": 1.0, "f_ref": 60.0, "mode": "normal"}

MATLAB side (simulink_tcp_client.m):
  tcpclient object on host="127.0.0.1" port=5760
  Send JSON string + newline every simulation step (0.1 s)
  Read response JSON back and apply to setpoint blocks.
"""

import json
import socket
import threading
import logging
import time

logger = logging.getLogger(__name__)

# Default bind address
HOST = "127.0.0.1"
PORT = 5760


# ---------------------------------------------------------------------------
# Import pipeline components (optional — graceful degradation if not present)
# ---------------------------------------------------------------------------
try:
    import sys, os
    _here = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(_here, ".."))
    from detection.feature_extraction import FeatureExtractor
    from digital_twin.synchronization  import DTSynchronizer
    from control.resilient_control     import ResilientController
    from detection.edge_inference      import EdgeInferenceEngine
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    logger.warning("Pipeline modules not found — bridge will echo dummy setpoints")


class CoSimBridge:
    """
    TCP server that accepts connections from the Simulink TCP client,
    processes each sensor measurement through the full Edge-AI pipeline,
    and returns updated setpoints.

    Parameters
    ----------
    model_path : str, optional
        Path to trained 1D-CNN weights. If None, uses random weights.
    host : str
        Bind address (default 127.0.0.1).
    port : int
        TCP port (default 5760, matching simulink_tcp_client.m).
    """

    def __init__(
        self,
        model_path: str = None,
        host: str = HOST,
        port: int = PORT,
    ):
        self.host = host
        self.port = port
        self._server: socket.socket = None
        self._running = False

        # Initialise pipeline if available
        if _PIPELINE_AVAILABLE:
            self.extractor  = FeatureExtractor(window_size=1)
            self.sync       = DTSynchronizer()
            self.controller = ResilientController()
            if model_path:
                self.engine = EdgeInferenceEngine(model_path)
            else:
                self.engine = None
                logger.warning("No model_path — inference disabled, controller uses DT state only")
        else:
            self.extractor = self.sync = self.controller = self.engine = None

    # ------------------------------------------------------------------
    def start(self):
        """Start the TCP server in a background thread."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(5)
        self._running = True
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()
        logger.info("CoSimBridge listening on %s:%d", self.host, self.port)
        return t

    def stop(self):
        self._running = False
        if self._server:
            self._server.close()

    # ------------------------------------------------------------------
    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._server.accept()
                logger.info("MATLAB client connected from %s", addr)
                threading.Thread(
                    target=self._handle_client,
                    args=(conn,),
                    daemon=True,
                ).start()
            except OSError:
                break

    def _handle_client(self, conn: socket.socket):
        buf = b""
        try:
            while self._running:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        msg = json.loads(line.decode())
                        resp = self._process(msg)
                        conn.sendall((json.dumps(resp) + "\n").encode())
                    except json.JSONDecodeError as e:
                        logger.error("JSON decode error: %s", e)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def _process(self, msg: dict) -> dict:
        """
        Run one pipeline step for a single MATLAB measurement message.

        msg keys: v, f, p, q, soc, irr, t
        """
        t0 = time.perf_counter()

        # Defaults
        v   = float(msg.get("v",   480.0))
        f   = float(msg.get("f",   60.0))
        p   = float(msg.get("p",   30000.0))
        q   = float(msg.get("q",   1000.0))
        soc = float(msg.get("soc", 60.0))
        irr = float(msg.get("irr", 800.0))

        measurement = {
            "voltage": v, "frequency": f,
            "active_p": p, "reactive_q": q,
            "soc_pct": soc, "irradiance": irr,
        }

        attack_class = 0
        confidence   = 1.0

        if _PIPELINE_AVAILABLE:
            # 1. DT sync & anomaly gate
            dt_state = self.sync.sync(measurement)

            # 2. Edge-AI inference (if model loaded)
            if self.engine is not None:
                import numpy as np
                feat = np.array([
                    v/480, f/60, p/1e5, q/1e4, soc/100, irr/1000,
                    *[0.0]*26   # remaining 26 features (placeholder)
                ], dtype="float32")
                result       = self.engine.predict(feat)
                attack_class = result.predicted_class
                confidence   = result.confidence

            # 3. Resilient control
            sp = self.controller.compute(
                attack_class, confidence,
                dt_state.voltage_pu, dt_state.frequency_hz, soc,
            )
        else:
            # Fallback: echo nominal setpoints
            from dataclasses import asdict
            sp = type("SP", (), {
                "pv_power_kw": 40, "bess_power_kw": 0,
                "diesel_power_kw": 40, "v_ref_pu": 1.0,
                "f_ref_hz": 60.0, "mode": "normal"
            })()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "pv_kw":      round(sp.pv_power_kw, 4),
            "bess_kw":    round(sp.bess_power_kw, 4),
            "diesel_kw":  round(sp.diesel_power_kw, 4),
            "v_ref":      round(sp.v_ref_pu, 6),
            "f_ref":      round(sp.f_ref_hz, 6),
            "mode":       sp.mode,
            "attack":     attack_class,
            "conf":       round(confidence, 4),
            "latency_ms": round(latency_ms, 4),
        }


# ---------------------------------------------------------------------------
# Companion MATLAB script (written to matlab/ folder on first run)
# ---------------------------------------------------------------------------
MATLAB_CLIENT_SCRIPT = r"""% simulink_tcp_client.m
% -------------------------------------------------------
% Connects to the Python CoSimBridge and exchanges JSON
% measurement / setpoint messages every simulation step.
%
% Usage: call from a MATLAB Function block or Model Callback.
% -------------------------------------------------------
function [pv_kw, bess_kw, diesel_kw, v_ref, f_ref] = ...
    cosim_step(v, f, p, q, soc, irr)

  persistent client
  if isempty(client)
    client = tcpclient('127.0.0.1', 5760, 'Timeout', 1);
  end

  msg = jsonencode(struct('v',v,'f',f,'p',p,'q',q,'soc',soc,'irr',irr));
  write(client, uint8([msg, newline]));

  % Wait for response (max 500 ms)
  tic;
  while client.NumBytesAvailable == 0 && toc < 0.5
    pause(0.001);
  end
  resp = jsondecode(char(read(client, client.NumBytesAvailable)));

  pv_kw     = resp.pv_kw;
  bess_kw   = resp.bess_kw;
  diesel_kw = resp.diesel_kw;
  v_ref     = resp.v_ref;
  f_ref     = resp.f_ref;
end
"""


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os, pathlib
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")

    # Write companion MATLAB script
    matlab_dir = pathlib.Path(__file__).parent.parent.parent / "matlab"
    matlab_dir.mkdir(parents=True, exist_ok=True)
    (matlab_dir / "simulink_tcp_client.m").write_text(MATLAB_CLIENT_SCRIPT)
    logger.info("MATLAB client script written to %s", matlab_dir / "simulink_tcp_client.m")

    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    bridge = CoSimBridge(model_path=model_path)
    t = bridge.start()

    print(f"CoSimBridge running on {HOST}:{PORT} — Ctrl-C to stop")
    try:
        t.join()
    except KeyboardInterrupt:
        bridge.stop()
        print("Bridge stopped.")
