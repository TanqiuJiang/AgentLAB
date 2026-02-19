"""
WebShop Subprocess Client
==========================
Drop-in replacement for WebShopEnvWrapper. Automatically spawns the
bridge process inside the webshop conda env. No server, no ports.

Usage:
    # In your ExperimentRunner.__init__, replace:
    #   self.env = WebShopEnvWrapper(config.num_products, config.verbose)
    # With:
    from webshop_client import WebShopSubprocessClient
    self.env = WebShopSubprocessClient(verbose=config.verbose)
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional


class WebShopSubprocessClient:
    """
    Same interface as WebShopEnvWrapper.
    Spawns webshop_bridge.py in the webshop conda env as a child process.
    Communicates via JSON lines over stdin/stdout pipes.
    """

    def __init__(self, conda_env: str = "webshop", num_products: int = None,
                 verbose: bool = True, bridge_script: str = None,
                 startup_timeout: int = 120):
        self.verbose = verbose
        self.conda_env = conda_env
        self._process: Optional[subprocess.Popen] = None

        # Locate the bridge script — it lives in the same directory as this file
        # Layout: envs/webshop/webshop_client.py
        #         envs/webshop/webshop_bridge.py
        if bridge_script:
            self._bridge_path = str(Path(bridge_script).resolve())
        else:
            self._bridge_path = str(Path(__file__).parent / "webshop_bridge.py")

        if not os.path.exists(self._bridge_path):
            raise FileNotFoundError(
                f"Bridge script not found: {self._bridge_path}\n"
                f"Expected at: envs/webshop/webshop_bridge.py"
            )

        # The bridge must run from the directory containing web_agent_site/
        self._bridge_cwd = str(Path(self._bridge_path).parent)

        # --- Public state (same as WebShopEnvWrapper) ---
        self.task_instruction: str = ""
        self.episode_reward: float = 0.0
        self.prices_seen: List[float] = []
        self.current_product_price: float = 0.0
        self.current_product_name: str = ""
        self.last_item_price: float = 0.0
        self.last_item_name: str = ""
        self.done: bool = False
        self.page_type: str = "search"
        self.num_tasks: int = 12087

        # Start the bridge
        self._start_bridge(num_products, startup_timeout)

    def _start_bridge(self, num_products: int, timeout: int):
        """Spawn the bridge process in the webshop conda env."""
        if self.verbose:
            print(f"Starting WebShop bridge in conda env '{self.conda_env}'...")

        cmd = [
            "conda", "run", "--no-capture-output", "-n", self.conda_env,
            "python", self._bridge_path,
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if not self.verbose else None,
                text=True,
                bufsize=1,  # line buffered
                cwd=self._bridge_cwd,  # MUST run from envs/webshop/ for imports
            )
        except FileNotFoundError:
            raise RuntimeError(
                "conda not found. Install Miniconda/Anaconda first.\n"
                "https://docs.conda.io/en/latest/miniconda.html"
            )

        # Send init config
        init_config = {}
        if num_products:
            init_config["num_products"] = num_products

        self._send(init_config)

        # Wait for ready signal
        if self.verbose:
            print("Waiting for WebShop environment to load...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self._recv(timeout=timeout)
            if response is None:
                self._check_process_alive()
                continue

            if "error" in response:
                raise RuntimeError(f"Bridge startup failed: {response['error']}")

            if response.get("status") == "ready":
                self.num_tasks = response.get("num_tasks", 12087)
                if self.verbose:
                    print(f"✓ WebShop bridge ready (num_tasks={self.num_tasks})")
                return

        raise TimeoutError(
            f"WebShop bridge did not start within {timeout}s.\n"
            f"Check that the '{self.conda_env}' conda env has WebShop installed."
        )

    def _send(self, data: dict):
        """Send a JSON line to the bridge."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Bridge process is not running")

        line = json.dumps(data) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

    def _recv(self, timeout: int = 60) -> Optional[dict]:
        """Read a JSON line from the bridge."""
        if self._process is None:
            raise RuntimeError("Bridge process is not running")

        # Simple blocking read (the bridge always responds with one line per command)
        line = self._process.stdout.readline()

        if not line:
            self._check_process_alive()
            return None

        line = line.strip()
        if not line:
            return None

        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # Might be a stray print statement; skip it
            if self.verbose:
                print(f"  [bridge noise] {line[:100]}")
            return None

    def _check_process_alive(self):
        """Check if the bridge process is still running."""
        if self._process and self._process.poll() is not None:
            rc = self._process.returncode
            stderr = ""
            if self._process.stderr:
                stderr = self._process.stderr.read()
            raise RuntimeError(
                f"Bridge process died (exit code {rc}).\n"
                f"Stderr: {stderr[:500] if stderr else '(not captured, run with verbose=True)'}"
            )

    def _call(self, data: dict) -> dict:
        """Send a command and get the response."""
        self._send(data)
        # Read lines until we get a valid JSON response
        for _ in range(10):  # skip up to 10 noise lines
            response = self._recv()
            if response is not None:
                if "error" in response:
                    raise RuntimeError(f"Bridge error: {response['error']}")
                return response
        raise RuntimeError("No response from bridge after 10 read attempts")

    # --- Public interface (same as WebShopEnvWrapper) ---

    def reset_state(self):
        self.task_instruction = ""
        self.episode_reward = 0.0
        self.prices_seen = []
        self.current_product_price = 0.0
        self.current_product_name = ""
        self.last_item_price = 0.0
        self.last_item_name = ""
        self.done = False
        self.page_type = "search"

    def reset(self, task_idx: int = None) -> str:
        self.reset_state()

        cmd = {"command": "reset"}
        if task_idx is not None:
            cmd["task_idx"] = task_idx

        data = self._call(cmd)

        obs = data["observation"]
        self.task_instruction = data.get("task_instruction", "")
        self.page_type = data.get("page_type", "search")

        for p in data.get("all_prices_on_page", []):
            if p > 0 and p not in self.prices_seen:
                self.prices_seen.append(p)

        return obs

    def step(self, action: str) -> Tuple[str, float, bool]:
        # Track pre-buy state (same as WebShopEnvWrapper)
        if action.lower() == "click[buy now]" and self.current_product_price > 0:
            self.last_item_price = self.current_product_price
            self.last_item_name = self.current_product_name

        data = self._call({"command": "step", "action": action})

        obs = data["observation"]
        reward = data.get("reward", 0.0)
        done = data.get("done", False)

        self.done = done
        self.episode_reward += reward
        self.page_type = data.get("page_type", "search")

        # Update prices
        for p in data.get("all_prices_on_page", []):
            if p > 0 and p not in self.prices_seen:
                self.prices_seen.append(p)

        # Update current product
        if self.page_type == "item":
            self.current_product_price = data.get("current_product_price", 0.0)
            self.current_product_name = data.get("current_product_name", "")
        else:
            self.current_product_price = 0.0
            self.current_product_name = ""

        # Post-buy update
        if done and action.lower() == "click[buy now]":
            if data.get("current_product_price", 0) > 0:
                self.last_item_price = data["current_product_price"]
            if data.get("current_product_name", ""):
                self.last_item_name = data["current_product_name"]

        return obs, reward, done

    def close(self):
        """Shut down the bridge process."""
        if self._process and self._process.poll() is None:
            try:
                self._send({"command": "quit"})
                self._process.wait(timeout=5)
            except:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except:
                    self._process.kill()
            if self.verbose:
                print("WebShop bridge stopped.")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()