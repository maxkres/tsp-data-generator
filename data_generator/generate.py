#!/usr/bin/env python3

import os
import sys

proj_root = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(proj_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tsp_generator.cli import main

if __name__ == "__main__":
    main()
