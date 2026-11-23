"""DEMO version of test_single_delay.py (5 s delay): same thing, with truck, truck-bay and forklift boxes drawn.
    python test_single_delay_demo.py --source CAM_URL    (all test_single_delay.py options work)"""
import sys

import test_single_delay

if __name__ == "__main__":
    if "--show-boxes" not in sys.argv:
        sys.argv.append("--show-boxes")
    sys.exit(test_single_delay.main())
