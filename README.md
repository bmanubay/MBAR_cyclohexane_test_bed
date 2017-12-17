`MBAR_var_CPU.py` uses `sys.argv` arguments, so call script using something like `python MBAR_var_CPU.py 0.1094 0.1126 1.9080 1.9140`. This will make a sparse grid of MBAR estimates over `epsilon` from 0.1094 to 0.1126 and `rmin_half` from 1.9080 to 1.9140 for the `[#6X4:1]` non-bonded SMIRKS type.

Lines `306` and `307` of `MBAR_var_CPU.py` specify the number of burnin samples for the trajectories. For quick testing purposes, I would recommend keeping them as is.
