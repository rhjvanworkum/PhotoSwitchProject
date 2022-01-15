from chemml.chem import Molecule
import os
import pandas as pd

orca_input_file_string = \
"""
! BP86 def2-SVP def2/J TightSCF
        
%maxcore 1000 # Memory settings often need to be modified when running TDDFT. Check batching info in the TDDFT output.
%tddft
nroots 150   # Setting the number of roots (transitions) to be calculated.
maxdim 5 # Davidson expansion space = MaxDim * nroots. Use MaxDim 5-10 for favorable convergence. Note that the larger MaxDim is, the more disk space is required
end

%scf MaxIter 150
end

* xyz   0   1
"""

def run_orca_calculation(orca_dir, smiles):
    if not os.path.isdir(orca_dir):
      os.makedirs(orca_dir)
      
    filename = orca_dir + smiles
    
    mol = Molecule(smiles, 'smiles')
    # add hydrogens
    mol.hydrogens('add')
    # optimize geometry
    mol.to_xyz(optimizer='MMFF', mmffVariant='MMFF94s', maxIters=300) # 'UFF'
    # save to xyz
    # atomic symbols
    symb = mol.xyz.atomic_symbols
    # atomic geometry
    geom = mol.xyz.geometry
    
    with open(filename+'.inp', 'w') as f:
        f.write(orca_input_file_string)
            
        for idx in range(len(symb)):
            symbol = symb[idx]
            geometry = geom[idx]
            f.write(' ' + str(symbol[0]) + '  ' + str(geometry)[1:-1])
            f.write('\n')
            
        f.write('\n')
        f.write('end')
        
    os.system("orca " + filename + '.inp' + " > " + filename + '.out')
    
def extract_wavelength_from_calculation(orca_dir, smiles, task):
  filename = orca_dir + smiles + '.out'
  
  if not os.path.exists(filename):
    return "Calculation has not been run yet"
  
  counter = 0
  temp_list = []
  with open(filename, 'r') as read:
    for line in read:
      if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
        for line in read:
          if line != '\n':
            counter += 1
            if counter >= 5:
              new_line = " ".join(line.split())
              tddft_vals = new_line.split()

              temp_list.append(tddft_vals)

          else:
            break
          
  df = pd.DataFrame(temp_list, columns=['State', 'Energy', 'Wavelength', 'fosc', 'T2', 'TX', 'TY', 'TZ'])
  df = df.sort_values(by=['fosc'], ascending=False)

  n_pi = list(df.iloc[0])
  pi_pi = list(df.iloc[1])
  
  if task == 'n_pi':
    return n_pi
  elif task == 'pi_pi':
    return pi_pi