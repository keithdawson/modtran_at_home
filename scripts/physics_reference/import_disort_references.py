"""Import upstream DISORT 4.0.99 fixtures and their exact PythonicDISORT inputs.

Run with the pinned Pythonic-DISORT clone as the sole argument. This executes
upstream test setup code but intercepts the solve call; no reference outputs
are generated using our implementation. Records source commit and file hashes.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[2]
clone = Path(sys.argv[1]).resolve()
provenance = json.loads((root/'mwir_atmos/_vendor/pythonic_disort/provenance.json').read_text())
assert subprocess.check_output(['git','-C',str(clone),'rev-parse','HEAD'], text=True).strip() == provenance['commit']
sys.path.insert(0, str(clone/'src'))
import PythonicDISORT

class Captured(Exception):
    pass

dest = root/'tests/reference_data/disort'
dest.mkdir(parents=True, exist_ok=True)
metadata = {'repository': provenance['repository'], 'commit': provenance['commit'],
            'outputs': 'Upstream Stamnes DISORT 4.0.99 fixtures, not recalculated.', 'cases': {}}
for case in ['1a','1d','2a','2d','7a','8a','9c']:
    file = clone/f'pydisotest/{case[0]}_test.py'
    spec = importlib.util.spec_from_file_location('upstream_case', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = {}
    def capture(*args, **kwargs):
        inputs.update(zip(('tau_arr','omega_arr','NQuad','Leg_coeffs_all','mu0','I0','phi0'), args))
        inputs.update(kwargs)
        raise Captured()
    PythonicDISORT.pydisort = capture
    try:
        getattr(module, 'test_'+case)()
    except Captured:
        pass
    assert inputs
    if 'BDRF_Fourier_modes' in inputs:
        # Selected test 9c uses a constant Lambertian albedo.
        modes = inputs.pop('BDRF_Fourier_modes')
        value = modes[0](np.array([.2,.6]), np.array([.3,.8]))
        assert len(modes) == 1 and np.all(value == value[0,0])
        inputs['surface_albedo'] = value[0,0]
    output = clone/f'pydisotest/Stamnes_results/{case}_test.npz'
    with np.load(output, allow_pickle=False) as expected:
        fields = {'expected_'+k: expected[k] for k in expected.files}
    fields.update({'input_'+k: v for k,v in inputs.items()})
    np.savez_compressed(dest/(case+'.npz'), **fields)
    metadata['cases'][case] = {'setup_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                               'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
(dest/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
(dest/'LICENSE.txt').write_text(next(clone.glob('LICENSE*')).read_text())
