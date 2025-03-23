import sys
from awfutils import MkSweep

assert len(sys.argv) == 2, "Usage: python sweep.py <sweepdir>"
sweepdir = sys.argv[1]

with MkSweep(sweepdir) as ms:
    cmd = "python train.py config/train_shakespeare_char.py "

    defaults = dict(
        dtype="bfloat16",
    )

    ms.add(cmd, defaults, qat="bfloat16")
    ms.add(cmd, defaults, qat="float16")

    for seed in range(5):
        ms.add(cmd, defaults, seed=seed + 42, qat_rnd="tne")
        for sr in ("sr", "srf", "srff"):
            for srnumbits in (1, 2, 3, 4, 8):
                ms.add(
                    cmd,
                    defaults,
                    seed=seed + 42,
                    qat="b8p4",
                    qat_rnd=sr,
                    qat_srn=srnumbits,
                )
