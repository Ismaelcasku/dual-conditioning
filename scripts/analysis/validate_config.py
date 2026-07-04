#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from dual_conditioning.config import load_campaign_config


parser = argparse.ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()
config = load_campaign_config(args.config)
print(json.dumps(asdict(config), indent=2))
