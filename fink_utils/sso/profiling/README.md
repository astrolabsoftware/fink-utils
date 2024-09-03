Profiling:

```
kernprof -l run_exp.py
python -m line_profiler -rmt "run_exp.py.lprof"
```

with default astropy:

```python
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   250         1          0.8      0.8      0.0      if phyparam is None:                                
   251                                                   # get the physical parameters with the latest d…
   252         1    2769024.4    3e+06     86.7          phyparam = extract_physical_parameters(pdf, fla…
   253                                                                                                   
   254                                               # Compute the residuals (obs - model)               
   255         1       9277.9   9277.9      0.3      residuals = compute_residuals(pdf, flavor, phyparam)
   256                                                                                                   
   257         2        162.0     81.0      0.0      model = LombScargleMultiband(                       
   258         1         40.0     40.0      0.0          pdf["i:jd"],                                    
   259         1          0.2      0.2      0.0          residuals,                                      
   260         1         32.0     32.0      0.0          pdf["i:fid"],                                   
   261         1         30.8     30.8      0.0          pdf["i:sigmapsf"],                              
   262         1          0.2      0.2      0.0          nterms_base=Nterms_base,                        
   263         1          0.2      0.2      0.0          nterms_band=Nterms_band,                        
   264                                               )                                                   
   265                                                                                                   
   266         2     403051.5 201525.7     12.6      frequency, power = model.autopower(                 
   267         1          0.3      0.3      0.0          method="fast",                                  
   268         1          0.3      0.3      0.0          sb_method=sb_method,                            
   269         1          1.2      1.2      0.0          minimum_frequency=1 / period_range[1],          
   270         1          0.3      0.3      0.0          maximum_frequency=1 / period_range[0],          
   271                                               )                                                   
```

With `fastnifty`:

```python
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   251         1          0.3      0.3      0.0      if phyparam is None:                                
   252                                                   # get the physical parameters with the latest d…
   253         1    2635831.0    3e+06     97.6          phyparam = extract_physical_parameters(pdf, fla…
   254                                                                                                   
   255                                               # Compute the residuals (obs - model)               
   256         1       8900.0   8900.0      0.3      residuals = compute_residuals(pdf, flavor, phyparam)
   257                                                                                                   
   258         2        132.6     66.3      0.0      model = LombScargleMultiband(                       
   259         1         39.2     39.2      0.0          pdf["i:jd"],                                    
   260         1          0.2      0.2      0.0          residuals,                                      
   261         1         32.6     32.6      0.0          pdf["i:fid"],                                   
   262         1         33.4     33.4      0.0          pdf["i:sigmapsf"],                              
   263         1          0.2      0.2      0.0          nterms_base=Nterms_base,                        
   264         1          0.2      0.2      0.0          nterms_band=Nterms_band,                        
   265                                               )                                                   
   266                                                                                                   
   267         2      51478.2  25739.1      1.9      frequency, power = model.autopower(                 
   268         1          0.4      0.4      0.0          method="fast",                                  
   269         1          0.3      0.3      0.0          sb_method=sb_method,                            
   270         1          0.9      0.9      0.0          minimum_frequency=1 / period_range[1],          
   271         1          0.3      0.3      0.0          maximum_frequency=1 / period_range[0],          
   272                                               )                      
```
