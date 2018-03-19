`MBAR_var_CPU.py` uses pickle files in the `pickles` directory in order to automatically set up MBAR objects and make estimates of H_vap and molar volumes based on two reference simulations. 

All of the observalbes have already been subsampled. The `u_kn` and `E_kn` matrices already reflect pressure contributions (i.e. they are reduced and unreduced enthalpies already). Simply call `python MBAR_car_CPU.py` to run. 
