set -eu
reference=/work/artifacts/physics-reference/MT_CKD_H2O
build=/work/artifacts/physics-reference/build
mkdir -p "$build"
cd "$build"
gfortran -fdefault-real-8 -fdefault-double-8 -std=legacy -I/usr/include $(nf-config --fflags) \
  "$reference/src/phys_consts.f90" "$reference/src/read_module.f90" \
  "$reference/src/mt_ckd_h2o_module.f90" /work/scripts/physics_reference/reference_cases.f90 \
  $(nf-config --flibs) -o reference_cases
cp "$reference/data/absco-ref_wv-mt-ckd.nc" .
./reference_cases
