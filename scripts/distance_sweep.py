"""Run a saved setup at increasing distances until band transmittance falls below a threshold.

Example: python scripts/distance_sweep.py mwir_setup.yaml --start .1 --stop 10 --step .5 --threshold .1
Replace the metric callback with your detector calculation when integrating this into another script.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import math
from mwir_atmos.setups import load_setup, sweep_distance
from mwir_atmos.spectral import band_average


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('setup',type=Path)
    parser.add_argument('--start',type=float,default=.1,help='First slant range in km')
    parser.add_argument('--stop',type=float,default=10.,help='Largest slant range in km')
    parser.add_argument('--step',type=float,default=.5,help='Distance increment in km')
    parser.add_argument('--threshold',type=float,default=.1)
    parser.add_argument('--band',type=float,nargs=2,metavar=('MIN_CM1','MAX_CM1'))
    parser.add_argument('--output',type=Path,default=Path('distance_sweep.csv'))
    args=parser.parse_args()
    if not all(math.isfinite(v) for v in [args.start,args.stop,args.step]) or args.start<=0 or args.stop<args.start or args.step<=0:
        parser.error('Distances must be finite, with 0 < start <= stop and step > 0')
    if not 0 <= args.threshold <= 1:
        parser.error('Transmittance threshold must be between 0 and 1')
    setup=load_setup(args.setup)
    band=tuple(args.band or (setup.wavenumber_min_cm1,setup.wavenumber_max_cm1))
    if not all(math.isfinite(v) for v in band) or not setup.wavenumber_min_cm1 <= band[0] < band[1] <= setup.wavenumber_max_cm1:
        parser.error('Band must be inside the saved spectral window')
    if math.floor((band[1]-setup.wavenumber_min_cm1)/setup.resolution_cm1)-math.ceil((band[0]-setup.wavenumber_min_cm1)/setup.resolution_cm1)<1:
        parser.error('Band must contain at least two spectral samples')
    count=math.floor((args.stop-args.start)/args.step)+1
    if count>10000:
        parser.error('Use at most 10000 distances; increase --step')
    distances=[args.start+i*args.step for i in range(count)]
    if not math.isclose(distances[-1],args.stop):
        distances.append(args.stop)
    with args.output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['distance_km','band_transmittance'])
        def record(sample):
            writer.writerow([sample.distance_km,sample.score])
            stream.flush()  # Keep completed samples if a later calculation fails.
        result=sweep_distance(setup,distances,threshold=args.threshold,
                              metric=lambda r:band_average(r.grid,r.transmittance.total,band),
                              include_radiance=False,on_sample=record)
    if result.crossed_threshold:
        print(f'First below-threshold sample: {result.first_failing_distance_km:g} km; '
              f'last passing sample: {result.last_passing_distance_km}')
    else:
        print('Threshold not crossed within the sampled distances.')
    print(f'Saved {len(result.samples)} samples to {args.output}')


if __name__=='__main__':
    main()
