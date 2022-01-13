parent_mols = [
  '[U](C1=CC=C(C=C1)N=NC2=CC=C([U])C=C2)' # azobenzene
  'C2(=CC=C([U])[N]2)N=NC3=CC=C([U])[N]3' # bisazopyrrole
  'C2(=CC=C([U])[S]2)N=NC3=CC=C([U])[S]3' # bisazothiophene
  'C2(=CC=C([U])[O]2)N=NC3=CC=C([U])[O]3' # bisazofuran
]

linkers = [
  '([H])',                                # H-terminus
  '(c2ccc([Y])cc2)',                      # benzene
  '(c2ncc([Y])cc2)',                      # pyridine
  '(c2ncc([Y])cn2)',                      # pyrimidine
  '(c2nnc([Y])nn2)',                      # tetrazine
  'C2=CC=C([Y])C2',                       # cyclopentadiene
  '(c2ccc([Y])N2)',                       # pyrrole (2,5) 
  '(c2cc([Y])cN2)',                       # pyrrole (2,4) 
  '(c2ccc([Y])N(C)2)',                    # pyrrole(N-methyl) 
  '(c2ccc([Y])N(C=O)2)',                  # pyrrole(N-COH)
  '(c1cnc([Y])N1)',                       # imidazole
  'c2ccc([Y])O2',                         # furan
  'c2ccc([Y])S2',                         # thiophene
  '(c2ccc([Y])S(=O)(=O)2)',               # thiophene(dioxide)
  '(c2sc([Y])cn2)',                       # thiazole (2,5)
  '(c2scc([Y])n2)',                       # thiazole (2,4)
  '(c1ncc([Y])o1)',                       # oxazole (2,5)
  '(c1nc([Y])co1)',                       # oxazole (2,4)
  '(C#C[Y])',                             # acetylene
  '/C=C/[Y]',                             # ethylene(trans)
  '(C=N[Y])'                              # imine
] 

terminals = [
  '([H])',                                # hydrogen
  '([OH])',                               # hydroxy 
  '[C](F)(F)F',                           # trifluoromethyl 
  '[O][C](F)(F)F',                        # trifluoromethoxy 
  '[C]',                                  # methyl
  '[O][C]',                               # methoxy 
  '[N+]([O-])=O',                         # nitro 
  '([SH])',                               # thiol 
  '[F]',                                  # fluoro 
  '[Cl]',                                 # chloro 
  'C#N'                                   # cyano 
]

from lib import generate_smiles

generate_smiles('library_01.txt', parent_mols, linkers, terminals)