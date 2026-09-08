"""Package independently generated AER outputs as small, offline test fixtures."""
from pathlib import Path
import subprocess
import numpy as np
from scipy.io import netcdf_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/physics-reference/MT_CKD_H2O'
COMMIT = '3e043b491a996a9679c65a45a57a3ec4317e9e79'
assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == COMMIT
assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip()
DEST = ROOT / 'tests/reference_data'
DEST.mkdir(parents=True,exist_ok=True)
a = np.genfromtxt(ROOT/'artifacts/physics-reference/build/mt_ckd_4_3_fortran.csv',delimiter=',',names=True)
np.savez_compressed(DEST/'mt_ckd_4_3_fortran.npz',**{n:a[n] for n in a.dtype.names})
with netcdf_file(SOURCE/'run_example/mt_ckd_h2o_output.nc',mmap=False) as f:
    values = {n:v.data.copy() for n,v in f.variables.items()}
np.testing.assert_array_equal(values['self_absorption'],a['self_cm2'][:107])
np.testing.assert_array_equal(values['frgn_absorption'],a['foreign_cm2'][:107])
np.savez_compressed(DEST/'mt_ckd_4_3_aer_example.npz',**values)
print('Packed five Fortran cases; case 1 matches the AER-distributed example exactly.')
