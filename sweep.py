import sys
from awfutils import MkSweep

assert len(sys.argv) == 2, "Usage: python sweep.py <sweepdir> < commands.txt"
sweepdir = sys.argv[1]

with MkSweep(sweepdir) as ms:
    for line in sys.stdin:
        ms.add(line.rstrip())

print("make -f", ms.makefile_path, "-j 2")
