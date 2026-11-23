"""DEMO version of test_single.py (no timers): same thing, with truck, truck-bay and forklift boxes drawn.
    python test_single_demo.py --source CAM_URL        (all test_single.py options work)"""
import sys

import test_single

if __name__ == "__main__":
    if "--show-boxes" not in sys.argv:
        sys.argv.append("--show-boxes")
    sys.exit(test_single.main())
